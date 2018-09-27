#!/bin/bash
scoring.sh rnn+ngram | awk 'FNR==2{print $0}FNR==13{print $0" 4-gram lattices - BASELINE"}'
scoring.sh rnn+ngram | awk 'FNR==3{print $0" n-best list - RNN"}'
scoring.sh rnn+ngram | awk 'FNR==8{print $0" n-best list - RNN+NGRAM"}' | head -n 1
scoring.sh adapt | awk 'FNR==8{print $0" n-best list, adapted - RNN+NGRAM"}' | head -n 1
