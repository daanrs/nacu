#!/usr/bin/env bash

mkdir lang1_scores
mkdir lang2_scores

for r in 1 4 9
do
  for f in lang1/*
  do
    java -jar negsel2.jar -self english.train -n 10 -r "$r" -c -l \
    < "$f" \
    > lang1_scores/"$r"_$(basename "$f")
  done

for f in lang2/*
do
  java -jar negsel2.jar -self english.train -n 10 -r 4 -c -l \
    < "$f" \
    > lang2_scores/$(basename "$f")
done
done

