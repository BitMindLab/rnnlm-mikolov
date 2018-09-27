#!/bin/bash

export LC_ALL=C
export LANG=C

rnnbin=rnn_rescore-0.1/rnn_rescore
rnnlm=rt09.bin # created by: rnn_rescore -convert rt09.rnn rt09.bin
rundir=1

mkdir -p $rundir

if [ -e $rnnlm ]; then
  echo "RNN model found";
else
  echo "converting RNN model";
  $rnnbin -convert rt09.rnn $rnnlm
fi

echo "create a baseline (using n-best and the language model/acoustic scores)"
export LANG=C;cat lm.prob | awk '{lms=0;for (i=1;i<=NF;i++){lms+=log($i)/log(10)};print lms}' > $rundir/lms
paste ams $rundir/lms nbest.txt | getbest.sh 32 -10 > $rundir/ngram.mlf

echo "baseline performance"
HResults -h -I rt07seval.ref.mlf /dev/null $rundir/ngram.mlf

echo "run rnn_rescoring"
time $rnnbin -nbest $rnnlm nbest.txt > $rundir/rnns

echo "rnn performance"
paste ams $rundir/rnns nbest.txt | getbest.sh 32 -10 > $rundir/rnn.mlf
HResults -h -I rt07seval.ref.mlf /dev/null $rundir/rnn.mlf

echo "performance rnn+ngram (linear interpolation of log scores)"
export LANG=C;paste $rundir/rnns $rundir/lms | awk -v lambda=0.5 '{print $1*lambda+$2*(1-lambda)}' > $rundir/rnn+ngram.s
paste ams $rundir/rnn+ngram.s nbest.txt | getbest.sh 32 -10 > $rundir/rnn+ngram.mlf

HResults -h -I rt07seval.ref.mlf /dev/null $rundir/rnn+ngram.mlf
