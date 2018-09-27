#!/bin/bash

awk -v LMS=$1 -v WIP=$2 '{totalScore=(NF-3)*WIP/2.718+$1+$2*LMS;$1=$2="";print $0,totalScore}' | sort |\
awk '
  BEGIN{maxhyp=-999999}
  {
    if (utt!=$1){ 
      if (FNR>1) print hyp; 
      utt=$1; 
      maxhyp=$NF;
      hyp=$0; 
    }else {
      if (maxhyp<$NF){
        maxhyp=$NF;
        hyp=$0
      }
    }
  }
  END{print hyp}' |\
awk 'BEGIN{print "#!MLF!#"}{print "\""$1".fea\"";print "<s>";for (i=2;i<NF;i++)print $i;print "</s>";print "."}' | sed "s/^'/\\\'/"
