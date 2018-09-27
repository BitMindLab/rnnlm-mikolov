#include "qrnnlm.h"
#include <QtCore>

#define VERSION "0.1"

void printHelp(QString msg) {
  if (!msg.isEmpty()) qDebug() << msg;
  qDebug() << "rnn_rescore" << VERSION <<"- (c) 2011 by Stefan Kombrink/katakombi@gmail.com";
  qDebug() << "based on rnnlm code (c) by Tom Mikolov/tmikolov@gmail.com";
  qDebug() << "usage:"; 
  qDebug() << "\t-help";
  qDebug() << "\t\tprints this help screen listing all use cases";
  qDebug() << "\t-benchmark model.bin n";
  qDebug() << "\t\tbenchmarks the computer for n seconds (default 20, if omitted) using RNN model model.bin";
  qDebug() << "\t-convert rnn.txt rnn.bin";
  qDebug() << "\t\tconverts a trained rnn model in txt format (rnnlm,v4/5) into binary format (rnn_rescore)";
  qDebug() << "\t-nbest rnn.bin nbest-file"; 
  qDebug() << "\t\tn-best list rescoring processing one hypothesis per line in the input file";
  qDebug() << "\t-lm-prob scores lambda-list";
  qDebug() << "\t\tcombine existing lm probabilities with rnn scores, use lambda-list for interpolation (default 1,0)";
  qDebug() << "\t\tfirst value:linear interpolation of probabilities, second value: linear interpolation of log scores";
  qDebug() << "\t-independent";
  qDebug() << "\t\tfor use with -nbest, restricts the length of used history to the start of each hypothesis";
  qDebug() << "\t\teffectively treating utterances as independent from each other.";
//  qDebug() << "\t\tslight speed-up (ca. 10%) at cost of slight loss in improvement (also ca. 10%)";  
}

QMap<QString,QString>params;

bool checkSwitch(QString p){
  return params.contains(p);
}

bool checkNoParam(QString p){
  if (checkSwitch(p)) return params[p].isEmpty();
  else return false;
}

bool checkOneParam(QString p,QVariant &v){
  bool r=false;
  if (params.contains(p)){
    QStringList pl=params[p].split(' ');
    if (pl.count()==1){
      r=true;
      v=pl[0];
    }
  }
  return r;
}

bool checkTwoParams(QString p,QVariant &v1,QVariant &v2){
  bool r=false;
  if (params.contains(p)){
    QStringList pl=params[p].split(' ');
    if (pl.count()==2){
      r=true;
      v1=pl[0];
      v2=pl[1];
    }
  }
  return r;
}

typedef enum {
  NoParams=0,
  Help,
  Convert,
  Benchmark,
  NBest,
  NBestLMS,
  Unknown=-1
} UseCase;

int main(int argc, char** argv){
  
  QString argName;

  // command line arguments can look like that:
  //
  // -debug   (just a switch)
  // -debug 2 (parameter with one value)
  // -convert text.rnn binary.rnn (parameter with multiple values
  //
  // order does not matter
  
  int i=1;
  while (i<=argc) {
    // name of the switch/param
    argName=QString(argv[i]);
    // arguments
    i++;
    while (i<=argc && !QString(argv[i]).startsWith('-')) { // while parameters pending continue parsing
      params[argName]=params[argName]+' '+argv[i];
      i++;
    } 
    params[argName]=params[argName].trimmed(); // remove eventually trailing white space
  }
  
  QRnnLM rnnlm;QVariant p1,p2;UseCase uc=Unknown;
  
  // determine use case
  
  // no parameters
  if (argc==1) uc=NoParams;
  // -help
  if (checkSwitch("-help")) uc=Help;
  // -benchmark
  if (checkSwitch("-benchmark")) uc=Benchmark;
  // -convert
  if (checkSwitch("-convert")) uc=Convert;
  // -nbest
  if (checkSwitch("-nbest")) uc=NBest;
  // -nbest+ngram
  if (checkSwitch("-nbest")&&checkSwitch("-lm-prob")) uc=NBestLMS;

  // process other options
  // -debug
  if (checkSwitch("-debug")) rnnlm.setDebugMode(-1); // enable ALL debug messages
  if (checkOneParam("-debug",p1)) rnnlm.setDebugMode(p1.toInt());
  
  bool hypCx=checkSwitch("-independent");

  int dur=20;QStringList ls;QVector<real>l;
  l.append(1);l.append(0); // use rnn and lmscore
  
  switch (uc) {
    case NoParams:
      printHelp("");
      break;  
    case Help:
      printHelp("Help screen requested, listing all use cases:");
      break;
    case Convert:
      if (checkTwoParams("-convert",p1,p2)) { // check if 'convert' has two parameters
        if (!rnnlm.loadRnn(p1.toString())) break;
        if (!rnnlm.writeRnnBin(p2.toString())) break;      
      };
      break;
    case Benchmark:
      checkOneParam("-benchmark",p1);
      if (checkTwoParams("-benchmark",p1,p2)) dur=p2.toInt();
      if (rnnlm.loadRnnBin(p1.toString())) rnnlm.benchmark(dur);
      break;
    case NBest:
      if (checkTwoParams("-nbest",p1,p2)) {
        if (!rnnlm.loadRnnBin(p1.toString())) break;
        if (!rnnlm.loadNbest(p2.toString())) break;
        if (hypCx) rnnlm.processNbestHypothesisContext(); else rnnlm.processNbestFullContext();
      } else {
        qDebug()<<"Need rnn.bin and nbest.txt";
      }
      break;
    case NBestLMS:
      if (checkTwoParams("-nbest",p1,p2)) {
        if (!rnnlm.loadRnnBin(p1.toString())) break;
        if (!rnnlm.loadNbest(p2.toString())) break;
      } else {
        qDebug()<<"Need rnn.bin and nbest.txt";
        break;
      }
      checkOneParam("-lm-prob",p1);
      if (checkTwoParams("-lm-prob",p1,p2)){
        ls=p2.toString().split(',');l.clear();
        for (int i=0;i<ls.size();i++) l.append(QVariant(ls[i]).toFloat());
      }
      rnnlm.setLambdas(l);
      if (!rnnlm.loadLMScores(p1.toString())) break;
      if (hypCx) rnnlm.processNbestHypothesisContext(); else rnnlm.processNbestFullContext();
      break;
    default:
      printHelp("Unable to recognize use case, listing all use cases:");
      break;
  }

  return 0;
}
