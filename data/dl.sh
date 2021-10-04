wget ftp://ftp.ircam.fr/pub/IRCAM/equipes/salles/listen/archive/SUBJECTS/IRC_1059.zip
unzip IRC_1059.zip
mkdir HRTF
mv COMPENSATED/WAV/IRC_1059_C/ HRTF/
rm -rf RAW/ COMPENSATED/ 
rm IRC_1059.zip
# convert to 16-bit 44.1 kHz
for f in $(find ./HRTF -name '*.wav'); do 
echo $f
ffmpeg -i $f -acodec pcm_s16le -ar 44100 ${f%.wav}z.wav; 
rm $f;
mv ${f%.wav}z.wav $f;
done

wget http://skaldir.bplaced.net/bilda/KalthallenCabsIR.rar
7z x KalthallenCabsIR.rar 
mkdir GtrCab
mv "KalthallenCabsIR/Kalthallen IRs" ./GtrCab
rm -rf KalthallenCabsIR
rm KalthallenCabsIR.zip

# convert to 16-bit 44.1 kHz
for f in $(find ./GtrCab -name '*.wav'); do 
echo $f
ffmpeg -i $f -acodec pcm_s16le -ar 44100 "${f%.wav}z.wav"; 
rm $f;
mv ${f%.wav}z.wav $f;
done