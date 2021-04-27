#!/bin/bash
mkdir HRTF
cd HRTF
for i in $(seq -f "%04g" 1002 1059)
do
  echo IRC_$i
  wget ftp://ftp.ircam.fr/pub/IRCAM/equipes/salles/listen/archive/SUBJECTS/IRC_$i.zip
  unzip IRC_$i.zip -d IRC_$i/
  rm IRC_$i.zip
  #cd IRC_$i/COMPENSATED/WAV
  #find . -name "*.wav" -exec ffmpeg -i {} -acodec pcm_s16le {}_16.wav -loglevel panic \;
  #cd ../../..
done

