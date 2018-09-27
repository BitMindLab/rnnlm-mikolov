#!/bin/bash

export LC_ALL=C
export LANG=C

AMS=ams
LMS=lm.prob
NLIST=nbest.txt
BASERNN=rt09.rnn
INMLF=results/rnn+ngram.lin.0.5.mlf

RNNLMBIN=rnnlm/rnnlm
RNNRESBIN=rnn_rescore-0.1/rnn_rescore

rundir=adapt
LR=0.1

mkdir -p $rundir

echo "this time an example of RNN adaptation (based on the output of example 2)"
echo "we take the 1-best and adapt on it, and run a second RNN rescoring"

echo "dump utterances per talk"
cat $INMLF | grep -v '<s>' | grep -v '</s>' | awk 'FNR>1{print}' | sed 's/"\*\///' | sed 's/\.fea"//' | awk '/^\.$/{print "";next}{printf $1" "}' | tr -d '"\\' | awk '{split($1,n,"_");print (n[3]+n[4])/2,$0 > "'$rundir'/"n[1]".txt"}'

echo "sort each talk by time"
for t in $rundir/*.txt ; do
  cat $t | sort -n -k 1 | awk '{$1=$2="";print $0}' | sed 's/^ *//' | sed 's/ *$//' > $t.timesorted
done

echo "adapt one model for each talk"
for t in $rundir/*.timesorted; do
  name=$(basename $t .txt.timesorted)
  cp $BASERNN $rundir/${name}-$LR.rnn
  echo "training model for talk $name..."
  $RNNLMBIN -rnnlm $rundir/${name}-$LR.rnn -one-iter -alpha $LR -train $t -debug 2
done

echo "convert all models"
for m in $rundir/*.rnn; do
$RNNRESBIN -convert $m $rundir/$(basename $m .rnn).bin
done

echo "rescore using all rnns"
# FIXME nbest list HAS TO BE IN TALK ORDER FOR THAT TO WORK
rm -f $rundir/RNN.scores $rundir/RNN.utterances
for m in $rundir/*.bin; do
  name=$(basename $m -${LR}.bin)
  echo "creating scores for talk $name"
  cat $NLIST | grep $name > $rundir/talk.nbest
  cat $NLIST | awk '{print $1}' | paste - $LMS | grep $name | awk '{$1="";print $0}' | sed 's/^ //' > $rundir/lms.tmp
  $RNNRESBIN -nbest $m $rundir/talk.nbest -lm-prob $rundir/lms.tmp 1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0 >> $rundir/RNN.scores
done
rm $rundir/talk.nbest $rundir/lms.tmp

echo "create mlfs / force-aligned mlfs /scoring"
mkdir -p $rundir/results
F=1
for l in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do
  cat $rundir/RNN.scores | awk '{print $(2*'$F'-1)}' > $rundir/mix.s
  paste $AMS $rundir/mix.s $NLIST | getbest.sh 32 -10 > results/adapt.lin.$l.mlf
  cat $rundir/RNN.scores | awk '{print $(2*'$F')}' > $rundir/mix.s
  paste $AMS $rundir/mix.s $NLIST | getbest.sh 32 -10 > results/adapt.linlog.$l.mlf
  let F=F+1
done
rm $rundir/mix.s

echo "scoring"
scoring.sh adapt
