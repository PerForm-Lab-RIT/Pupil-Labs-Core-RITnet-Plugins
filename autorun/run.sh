#!/bin/bash

# Kevin Barkevich
# [run.sh <csvfile> <pupil>] - Parses a csv file <csvfile> and runs an instance of
#				Pupil Labs Core (located in the folder <pupil>) for each line with
#				settings based on that line
# 5/24/2021

IN_FILE=$1		# arg 1 - csv file
PUPIL_LOCATION=$2
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

run_pupil () {
	NAME=$1
	VIDEOFOLDER=$2
	PLUGIN=$3
	COUNT=$4
	PLUGIN=$(echo $PLUGIN | tr -d '\r')
	NEWDIR="$SCRIPT_DIR/$COUNT - $NAME"
	
	cp -r $VIDEOFOLDER $NEWDIR
	
	old="$(pwd)"
	cd "$PUPIL_LOCATION/pupil_src"
	
	EXECUTE="python main.py player \"$NEWDIR\" --plugin=$PLUGIN"
	python "main.py" "player" "$NEWDIR" --plugin=$PLUGIN &
	PROCESS=$!
	sleep 14
	echo $PROCESS
	echo "JOBS"
	jobs -l
	echo "WAITING FOR JOB TO FINISH"
	while [ ! -f "$NEWDIR/offline_data/offline_pupil.meta" ]
	do
		sleep 0.2
	done
	echo "KILLING $PROCESS"
	kill -9 $PROCESS
	sleep 5
	jobs -l
	wait
	cd "$old"
}

total=$(wc -l $IN_FILE)
count=1
{
	read
	while IFS=, read -r line; do
		# ...
		oIFS="$IFS"
		IFS="," read -ra ADDR <<< "$line"
		run_pupil "${ADDR[0]}" "${ADDR[1]}" "${ADDR[2]}" "$count"
		IFS="$oIFS"
		echo "$count / $total"
		[[ "$line" == *","* ]] && ((++count))
	done
} < $IN_FILE; \

IFS="," read -ra ADDR <<< "$line"
run_pupil "${ADDR[0]}" "${ADDR[1]}" "${ADDR[2]}"
echo "$count / $total"

echo ">> we found ${count} lines"
sleep infinity