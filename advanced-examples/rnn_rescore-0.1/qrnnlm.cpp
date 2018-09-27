#include "qrnnlm.h"

#include <QtCore>
#include <math.h>

#define P(w) isIV(w)? (y(w)*c(int2class[w])*c_norm*y_norm):(0)

bool QRnnLM::loadRnn(QString rnnFileName) {
  QFile rnnFile(rnnFileName);

  if (!rnnFile.open(QIODevice::ReadOnly)) {
    debug(QString("Can not open RNN model file: %1 ").arg(rnnFileName));
    return false;
  }

  QString junk;
  int ver,layer0Size,layer1Size,layer2Size,vocabSize;
  QTextStream rnns;
  int lcnt=0;
  QMap<int,QVariant> val;

  rnns.setDevice(&rnnFile);
  
  // read header (what is important for evaluating model)
  ver=rnns.readLine().split(':').last().toInt();
  
  int hdrcnt=22;
  if (ver==5) hdrcnt++; // ver 5 has bptt_block
  while(!rnns.atEnd()&&lcnt!=hdrcnt) {
    QString line=rnns.readLine().split(':').last();
    
    debug(QString("%1 %2").arg(line).arg(lcnt),1);
    val[lcnt]=line;
    lcnt++;
  }
  
  layer0Size=val[10].toInt();layer1Size=val[11].toInt();layer2Size=val[12].toInt();
  bptt=val[13].toInt();
  
  if (ver==4) {
    vocabSize=val[14].toInt();classSize=val[15].toInt();
  } else if (ver==5) {
    bptt_block=val[14].toInt();
    vocabSize=val[15].toInt();classSize=val[16].toInt();
  } else {
    vocabSize=0;
    debug("Error: Unknown RNN Version Format!");
  }
  
  // read vocabulary/classes
  vocab.clear();
  word2int.clear();
  int2word.clear();
  int2class.clear();
  int wix=0;
  for (int w=0;w<vocabSize;w++) {
    QStringList cols=QString(rnns.readLine()).split('\t');
    debug(cols[2]+"\n",0);
    QString wd=cols[2];
    int cl=QVariant(cols[3]).toInt();
    vocab.insert(wd);
    word2int[wd]=wix; // mapping word string -> int
    int2word[wix]=wd; // mapping int -> word string
    int2class[wix]=cl; // mapping int -> class
    if (!class2minint.contains(cl)) class2minint[cl]=wix; // mapping class -> start int
    class2maxint[cl]=qMax<int>(wix,class2maxint[cl]); // mapping class -> max int
    wix++;
  }
  
  // read hidden layer activations
  rnns.readLine();
  rnns.readLine();
  s=VectorXr(layer1Size);y=VectorXr(layer2Size-classSize);
  for (int i=0;i<layer1Size;i++) s(i)=QVariant(rnns.readLine()).toFloat();
  rnns.readLine();
  rnns.readLine();
  
  // read weights 0-1
  MatrixXr syn0=MatrixXr(layer1Size,layer0Size);
  for (int i=0;i<layer1Size;i++) 
    for (int j=0;j<layer0Size;j++) {
      syn0(i,j)=rnns.readLine().toFloat();
    }
  rnns.readLine();rnns.readLine();rnns.readLine();

  // read weights 1-2
  MatrixXr syn1=MatrixXr(layer2Size,layer1Size);
  for (int i=0;i<layer2Size;i++) 
    for (int j=0;j<layer1Size;j++) {
      syn1(i,j)=rnns.readLine().toFloat();
    }
    
  // decompose into four sub-matrices (for easier access!) 
  T=syn0.rightCols(s.size());
  U=syn0.leftCols(vocabSize);
  V=syn1.topRows(vocabSize);
  W=syn1.bottomRows(classSize);

  // done!
  rnnFile.close();
  
  return true;
}

bool QRnnLM::loadRnnBin(QString fn){
  QFile rnn(fn);
  if (rnn.open(QIODevice::ReadOnly)){
    QDataStream ds(&rnn);
    ds.setVersion(QDataStream::Qt_4_6);
    ds.setFloatingPointPrecision(QDataStream::SinglePrecision);
    int neu0_size,neu1_size,neu2_size,vocab_size;
    ds>>neu0_size>>neu1_size>>neu2_size>>vocab_size>>classSize;
    ds>>bptt>>vocab>>int2word>>word2int>>int2class>>class2minint>>class2maxint;
    s.resize(neu1_size);y.resize(vocab_size);c.resize(classSize);
    
    MatrixXr syn0, syn1; syn0.resize(neu1_size,neu0_size);syn1.resize(neu2_size,neu1_size);
    int i,j;
    for (i=0;i<s.size();i++) ds>>s(i); // restore s(t-1)
    for (i=0;i<neu1_size;i++)
      for (j=0;j<neu0_size;j++)
        ds>>syn0(i,j);
        
    for (i=0;i<neu2_size;i++)
      for (j=0;j<neu1_size;j++)
        ds>>syn1(i,j);
        
    rnn.close();
    // decompose into four sub-matrices (for easier access!) 
    T=syn0.rightCols(s.size());U=syn0.leftCols(vocab_size);
    V=syn1.topRows(vocab_size);W=syn1.bottomRows(classSize);
  }else {
    debug(QString("Can not load RNN binary file: %1").arg(fn));
    return false;
  }
  return true;
}

bool QRnnLM::writeRnnBin(QString fn){
  QFile rnn(fn);
  if (rnn.open(QIODevice::WriteOnly)){
    QDataStream ds(&rnn);
    int neu0_size,neu1_size,neu2_size,vocab_size;
    vocab_size=vocab.size();
    neu0_size=s.size()+vocab.size();neu1_size=s.size();neu2_size=vocab_size+classSize;
    
    ds.setVersion(QDataStream::Qt_4_6);
    ds.setFloatingPointPrecision(QDataStream::SinglePrecision);
    ds<<neu0_size<<neu1_size<<neu2_size<<vocab_size<<classSize;
    ds<<bptt<<vocab<<int2word<<word2int<<int2class<<class2minint<<class2maxint;
    
    MatrixXr syn0, syn1; syn0.resize(neu1_size,neu0_size);syn1.resize(neu2_size,neu1_size);
    syn0.rightCols(s.size())=T;syn0.leftCols(vocab.size())=U;
    syn1.topRows(vocab.size())=V;syn1.bottomRows(classSize)=W;
    
    int i,j;
    for (i=0;i<s.size();i++) ds<<s(i); // save s(t-1)
    for (i=0;i<neu1_size;i++)
      for (j=0;j<neu0_size;j++)
        ds<<syn0(i,j);
    for (i=0;i<neu2_size;i++)
      for (j=0;j<neu1_size;j++)
        ds<<syn1(i,j);
    rnn.close();
  }else {
    debug(QString("Can not write RNN binary file: %1").arg(fn));
    return false;
  }
  return true;
}

bool QRnnLM::loadNbest(QString nbestFileName) {
  QFile nb(nbestFileName);
  if (!nb.open(QIODevice::ReadOnly|QIODevice::Text)) {
    debug(QString("ERROR: could not open nbest list %1 for reading!").arg(nbestFileName));
    return false;
  };
  
  // iterate over all hypotheses (lines) in the file
  int nOOV=-1;
  QStringList hyp;
  
  while (!nb.atEnd()){
    hyp=QString(nb.readLine()).trimmed().split(' ');
    nBestHypLength.append(hyp.size()-1);
    nBestHypName.append(hyp[0]);
    for (int w=1;w<hyp.size();w++) {
      if (word2int.contains(hyp[w])) 
        nBestList.append(word2int[hyp[w]]);
      else {
        nOOV--;
        nBestList.append(nOOV);
        word2int[hyp[w]]=nOOV;
        int2word[nOOV]=hyp[w];
      }
    }
  }
  nb.close();
  nBestList.squeeze();
  debug(QString("Read %1 nbest hypotheses / %2 words").arg(nBestHypName.size()).arg(nBestList.size()),1);
  return true;
}

void QRnnLM::processNbestFullContext(){
  const int precision=8;
  const int hiddenSize=s.size();
  const int vocabSize=vocab.size();
  const double oovp=1.0/vocabSize;
  const double oovlp=log10(oovp);
  const bool hasLMScores=wordHypLMScores.size()!=0;
  
  QTextStream out(&debugFile);

  VectorXr context(hiddenSize);
  VectorXr context2(hiddenSize);
  QVector<double> linLambda,linlogLambda;
  
  int w=0, lastW=0,firstNbest=true;
  int wcnb=0,wclm=0;
  double lms,p=0, wordp=0; // you really need high precision here!

  s.setZero(); // clear hidden layer
  computeNet(0, 0); // compute an initial context in the hidden layer 
  context=context2=s; // store it
  
  QString hypName; // unique name in the hypothesis of the current nbest
  for (int u=0;u<nBestHypName.size();u++) {
    // this is the very first nbest of an hypothesis!
    if (hypName != nBestHypName[u]) {
      hypName = nBestHypName[u];
      firstNbest=true;
      context=s; // save the current context
      s=context2; // now set the previously saved context...
    } else s=context; // ...or otherwise reset to initial context!
    
    // process words
    wordp=0;
    linLambda.fill(0,lambdas.size());
    linlogLambda.fill(0,lambdas.size());
    for (int v=0;v<nBestHypLength[u];v++) {
      w=nBestList[wcnb];
      computeNet(lastW,w);
      if (isIV(w)) {
        p=y(w)*c(int2class[w])*c_norm*y_norm; // joint probability
        wordp+=log10(p);
      } else {
        p=oovp;
        wordp+=oovlp;
      }
      // interpolate linearly / linear interpolation of log scores
      // in case external scores are provided
      if (hasLMScores) {
        lms=wordHypLMScores[wclm]; 
        for (int l=0;l<lambdas.size();l++) {
          if (lms==0) {
            debug(QString("Unknown word %1").arg(int2word[w]),1);
            lms=oovp;
          }
          if (isOOV(w)) p=lms;
          linLambda[l]+=log10(lambdas[l]*p+(1.0-lambdas[l])*lms);
          linlogLambda[l]+=log10(p)*lambdas[l]+log10(lms)*(1-lambdas[l]);
        }
      }
      lastW=w;
      wcnb++;wclm++;
    }
    computeNet(lastW,0); // end of sentence
    p=y(0)*c(int2class[0])*c_norm*y_norm;
    wordp+=log10(p);
    if (hasLMScores) {
      lms=wordHypLMScores[wclm]; 
      for (int l=0;l<lambdas.size();l++) {
        if (lms==0) debug("ERROR:SCORE-NBEST MISMATCH!");
        linLambda[l]+=log10(lambdas[l]*p+(1.0-lambdas[l])*lms);
        linlogLambda[l]+=log10(p)*lambdas[l]+log10(lms)*(1-lambdas[l]);
      }
    }
    lastW=0;
    wclm++;
    // iterate over all available lambdas!!!
    for (int l=0;l<lambdas.size();l++) {
      out<<QString::number(linLambda[l],'g',precision)<<" "<<QString::number(linlogLambda[l],'g',precision);
      if (l<lambdas.size()-1) out<<" ";
    }
    if (lambdas.empty()) {out<<QString::number(wordp,'g',precision);}
    out<<"\n";
    
    // after processing the first nbest of a new utterance, save this context for the next utterance
    if (firstNbest) {
      context=context2;
      context2=s;
      firstNbest=false;
    }
  }
}

bool QRnnLM::loadLMScores(QString fn) {
  debug("loading lm scores now...",1);
  QFile ngs(fn);
  
  if (!ngs.open(QIODevice::ReadOnly|QIODevice::Text)) {
    debug(QString("Cannot load ngram score file: %1").arg(fn));
    return false;
  }
  
  wordHypLMScores.clear(); // delete previous ngram scores
  QString tmp;
  QStringList hyp;
  while (!ngs.atEnd()) {
    tmp=ngs.readLine();
    hyp=tmp.trimmed().split(' ');
    for (int n=0;n<hyp.size();n++) {
      wordHypLMScores.append(hyp[n].toFloat());
    }
  }
  
  wordHypLMScores.squeeze();
  ngs.close();
  return true;
}

void QRnnLM::computeNet(int lastW, int w) {
  debug(QString("CN %3 | %2 -- %1").arg(P(lastW)).arg(int2word[lastW]).arg(int2word[w]),1);
  VectorXr s_ac; // temporary variables for helping optimization
  
  s_ac.noalias()=-T*s; // s(t)=-T*s(t-1)
  if (isIV(lastW)) s_ac-=U.col(lastW); // s(t)=-s(t)-U*w-1(t)
    
  // activate hidden layer (sigmoid) and determine updated s(t)
  s.noalias()=VectorXr(VectorXr(s_ac.array().exp()+1.0).array().inverse());
  
  if (isIV(w)){
    // evaluate classes: c(t)=W*s(t) + activation class layer (softmax)
    c.noalias()=VectorXr((W*s).array().exp()); c_norm=1.0/c.sum();
    // evaluate post. distribution for all words within that class: y(t)=V*s(t)
    int b=class2minint[int2class[w]];
    int n=class2maxint[int2class[w]]-b+1;

    // determine distribution of class of the predicted word
    // activate class part of the word layer (softmax)
    y.segment(b,n).noalias()=VectorXr((V.middleRows(b,n)*s).array().exp());
    y_norm=1.0/y.segment(b,n).sum();
  }
}

void QRnnLM::processNbestHypothesisContext(){
  const int precision=8;
  const int vocabSize=vocab.size();
  const double oovp=1.0/vocabSize;
  const double oovlp=log10(oovp);
  const bool hasLMScores=wordHypLMScores.size()!=0;

  QTextStream out(&debugFile);
  QVector<double> linLambda,linlogLambda;

  int w, lastW;
  int wcnb=0,wclm=0;
  double lms=0,p=0, wordp=0; // you really need high precision here!

  s.setZero(); // clear hidden layer state
  computeNet(0, 0); // reset net
  lastW=0;
  VectorXr reset=s;// store reset state

  for (int u=0;u<nBestHypName.size();u++) {
    // process words
    wordp=0;
    linLambda.fill(0,lambdas.size());
    linlogLambda.fill(0,lambdas.size());

    for (int v=0;v<nBestHypLength[u];v++) {
      w=nBestList[wcnb];
      computeNet(lastW,w);
      if (isIV(w)) {
        p=y(w)*c(int2class[w])*c_norm*y_norm;
        wordp+=log10(p);
      } else {
        p=oovp;
        wordp+=oovlp;
      }

      // interpolate linearly / linear interpolation of log scores
      if (hasLMScores) {
        lms=wordHypLMScores[wclm];
        for (int l=0;l<lambdas.size();l++) {
          if (lms==0) {
            debug(QString("Unknown word %1").arg(int2word[w]),1);
            lms=oovp;
          }
          if (isOOV(w)) p=lms;
          linLambda[l]+=log10(lambdas[l]*p+(1.0-lambdas[l])*lms);
          linlogLambda[l]+=log10(p)*lambdas[l]+log10(lms)*(1-lambdas[l]);
        }
      }
      lastW=w;
      wcnb++;wclm++;
    }
    computeNet(lastW,0); // end of sentence
    p=y(0)*c(int2class[0])*c_norm*y_norm;
    wordp+=log10(p);
    if (hasLMScores) lms=wordHypLMScores[wclm];
    for (int l=0;l<lambdas.size();l++) {
      if (lms==0) debug("ERROR:SCORE-NBEST MISMATCH!");
      linLambda[l]+=log10(lambdas[l]*p+(1.0-lambdas[l])*lms);
      linlogLambda[l]+=log10(p)*lambdas[l]+log10(lms)*(1-lambdas[l]);
    }
    lastW=0;
    wclm++;
    // iterate over all available lambdas!!!
    for (int l=0;l<lambdas.size();l++) {
      out<<QString::number(linLambda[l],'g',precision)<<" "<<QString::number(linlogLambda[l],'g',precision);
      if (l<lambdas.size()-1) out<<" ";
    }
    if (lambdas.empty()) {out<<QString::number(wordp,'g',precision);}
    out<<"\n";

    // after processing the first nbest of a new utterance, reset to initial context!
    s=reset;
  }
}


void QRnnLM::benchmark(int seconds) {
  QTime t;
  t.start();
  int c=0;
  s.setZero();
  while (t.elapsed()<1000*seconds){
    for (int d=0;d<1000;d++) computeNet(0,0);c++;
  }
  debug(QString("Benchmarked %1 sec -- avg. computeNet()/sec: %2").arg(0.001*t.elapsed()).arg(1000000.0/t.elapsed()*c));
}
