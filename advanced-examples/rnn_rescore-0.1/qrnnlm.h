#ifndef QRNNLM_H
#define QRNNLM_H

#include <QtCore>

#include "Eigen/Dense"
using namespace Eigen;

// can be changed to float for small models (trained on less than 10M words)
// and for evaluation of models, else choose double
typedef float real;
typedef VectorXf VectorXr;
typedef MatrixXf MatrixXr;

class QRnnLM : public QObject {
  private:
  // files to handle
  QFile debugFile;
  
  // for storing nbest lists and scores
  QVector<QString> nBestHypName;
  QVector<int> nBestHypLength;
  QVector<int> nBestList;
  //QVector<real> nBestHypACScores;
  QVector<real> wordHypLMScores;
  
  // for rescoring
  qreal lms,wip;
  
  int randSeed,debugMode,version;
  
   // values for model interpolation
  QVector<real> lambdas;
  
  // RNN
  int bptt,bptt_block;
  QSet<QString> vocab;
  
  // map integers to words (IV/OOV) and vice versa
  QMap<int,QString> int2word;
  QMap<QString,int> word2int;
  QMap<int,int> int2class;
  QMap<int,int> class2minint;
  QMap<int,int> class2maxint;
  int classSize;
  
  VectorXr s,c,y; // hidden/class/output layer
  real c_norm,y_norm; // normalization for the sigmoids
  MatrixXr T,U,V,W; // recurrent/input/output/class weights
  
  protected:
    void debugnb(QString s, int mode=-1){
      QTextStream ds(&debugFile);
      if (mode<0) {
        ds<<s;
        return;
      }
      if (debugMode & mode) ds<<s;
    }
    
    void debug(QString s, int mode=-1) {
      debugnb(s+"\n",mode);
    }
    
    bool isOOV(int w) const {
      return w<=-1;
    }
    
    bool isIV(int w) const {
      return !isOOV(w);
    }

  public:
    QRnnLM(QObject* parent=NULL) : QObject(parent) {
      setDebugMode();
      debugFile.open(stdout,QIODevice::WriteOnly);
    }
    virtual ~QRnnLM() {};
    
    bool loadRnn(QString rnnFile);
    bool loadRnnBin(QString rnnFile);
    bool writeRnnBin(QString rnnFile);
    bool loadNbest(QString nbestFile);
    void processNbestFullContext();
    void processNbestHypothesisContext();
    bool loadLMScores(QString lmScoreFile);
    
    void computeNet(int lastW,int w);
    
    void setDebugMode(int d=0){
      debugMode=d;
    }
    
    void setLambdas(QVector<real> l) {
      lambdas=l;
    }
    void benchmark(int seconds=20);
};

#endif
