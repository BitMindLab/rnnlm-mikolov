#!/bin/bash
echo "LIN INTERPOLATION"
echo "lambda  |           # Snt |  Corr    Sub    Del    Ins    Err  S. Err |"
for l in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do
  HResults -h -I rt07seval.ref.mlf /dev/null results/$1.lin.$l.mlf|grep "Sum/Avg" | awk '{print "'$l'",$0}'
done
echo "LIN INTERPOLATION OF LOG SCORES"

echo "lambda  |           # Snt |  Corr    Sub    Del    Ins    Err  S. Err |"
for l in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do
  HResults -h -I rt07seval.ref.mlf /dev/null results/$1.linlog.$l.mlf|grep "Sum/Avg" | awk '{print "'$l'",$0}'
done
