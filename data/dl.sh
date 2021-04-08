#!/bin/bash
mkdir HRTF
for i in $(seq -f "%04g" 1002 1059)
do
  wget ftp://ftp.ircam.fr/pub/IRCAM/equipes/salles/listen/archive/SUBJECTS/IRC_$i.zip
  unzip IRC_$i.zip -d IRC_$i/
  mv IRC_$i/ HRTF/IRC_$i
  rm IRC_$i.zip
done

