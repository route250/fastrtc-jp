#!/bin/bash
#

REPO=https://github.com/VOICEVOX/voicevox_blog.git

WORK=tmp/voicevox_data/xyz
CONTENTS=tmp/voicevox_data/cache/contents
FILES=tmp/voicevox_data/cache/files
ls -ld $CONTENTS
mkdir -p $WORK
git clone $REPO $WORK
(cd $WORK; git pull )

AUDIO_DIR=src/assets/talk-audios
mkdir -p $CONTENTS/$AUDIO_DIR/
rm -rf $CONTENTS/$AUDIO_DIR/*.wav
cp -p $WORK/$AUDIO_DIR/*.wav $CONTENTS/$AUDIO_DIR/
mkdir -p $(dirname $FILES/$AUDIO_DIR)
(cd $CONTENTS/$AUDIO_DIR; ls -1 *.wav) | jq -R -s -c 'split("\n")[:-1]' > $FILES/$AUDIO_DIR

AUDIO_DIR=src/assets/dormitory-audios
mkdir -p $CONTENTS/$AUDIO_DIR/
rm -rf $CONTENTS/$AUDIO_DIR/*.wav
cp -p $WORK/$AUDIO_DIR/*.wav $CONTENTS/$AUDIO_DIR/
mkdir -p $(dirname $FILES/$AUDIO_DIR)
(cd $CONTENTS/$AUDIO_DIR; ls -1 *.wav) | jq -R -s -c 'split("\n")[:-1]' > $FILES/$AUDIO_DIR

CHAR_DIR=src/constants/characterInfos
mkdir -p $CONTENTS/$CHAR_DIR/
rm -rf $CONTENTS/$CHAR_DIR/*.ts
cp -p $WORK/$CHAR_DIR/*.ts $CONTENTS/$CHAR_DIR/
mkdir -p $(dirname $FILES/$CHAR_DIR)
(cd $CONTENTS/$CHAR_DIR; ls -1 *.ts) | jq -R -s -c 'split("\n")[:-1]' > $FILES/$CHAR_DIR
