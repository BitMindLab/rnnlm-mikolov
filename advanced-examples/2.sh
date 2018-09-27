#!/bin/bash

export LC_ALL=C
export LANG=C

rnnbin=rnn_rescore-0.1/rnn_rescore
rnnlm=rt09.bin
rundir=2

mkdir -p $rundir

if [ -e $rnnlm ]; then
  echo "RNN model found";
else
  echo "converting RNN model";
  $rnnbin -convert rt09.rnn $rnnlm
fi

echo "this time an example where a wide range of interpolation values are used"
echo "rescoring performance is NOT affected by that"

echo "creating all scores..."
time $rnnbin -nbest $rnnlm nbest.txt -lm-prob lm.prob 1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0 > $rundir/rnn.all.s

echo "extracting one-bests"
mkdir -p results
F=1
for l in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do
  cat $rundir/rnn.all.s | awk '{print $(2*'$F'-1)}' > $rundir/mix.s
  paste ams $rundir/mix.s nbest.txt | getbest.sh 32 -10 > results/rnn+ngram.lin.$l.mlf
  cat $rundir/rnn.all.s | awk '{print $(2*'$F')}' > $rundir/mix.s
  paste ams $rundir/mix.s nbest.txt | getbest.sh 32 -10 > results/rnn+ngram.linlog.$l.mlf
  let F=F+1
done
rm $rundir/mix.s

scoring.sh rnn+ngram
