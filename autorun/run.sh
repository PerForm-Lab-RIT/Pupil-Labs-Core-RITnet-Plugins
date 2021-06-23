#!/bin/bash

# Kevin Barkevich
# [run.sh <csvfile> <pupil>] - Parses a csv file <csvfile> and runs an instance of
#				Pupil Labs Core (located in the folder <pupil>) for each line with
#				settings based on that line
# 5/24/2021

VERBOSE=0
while getopts ":hv" OPTION; do
	case ${OPTION} in
		h )
			echo "Usage:"
			echo "	run.sh -h"
			echo "		Display this help message."
			echo
			echo "	run.sh [-v] <in_file> <pupil_folder>"
			echo "		Parse <in_file> csv and run each line's specified"
			echo "		configuration in the pupil install at <pupil_folder>."
			echo "			[-v]: Verbose"
			exit
			;;
		v )
			VERBOSE=1
			;;
		\? )
			;;
	esac
done
shift $((OPTIND -1))

IN_FILE=$1		# arg 1 - csv file
PUPIL_LOCATION=$2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

run_pupil () {
	NAME=$1
	VIDEOFOLDER=$2
	PLUGIN=$3
	COUNT=$4
	SAVE_EYE0=$5
	SAVE_EYE1=$6
	CUSTOMELLIPSE=$7
	MINIMUMDIAMETER=$8
	
	if [ "$SAVE_EYE0" = "true" ] && [ "$SAVE_EYE1" = "true" ]
	then
		SAVE_MASKS="both"
	elif [ "$SAVE_EYE0" = "true" ]
	then
		SAVE_MASKS="0"
	elif [ "$SAVE_EYE1" = "true" ]
	then
		SAVE_MASKS="1"
	else
		SAVE_MASKS="none"
	fi
	
	if [ "$CUSTOMELLIPSE" = "true" ]
	then
		CUSTOMELLIPSE="--custom-ellipse"
	else
		CUSTOMELLIPSE=""
	fi
	
	PLUGIN=$(echo $PLUGIN | tr -d '\r')
	NEWDIR="$SCRIPT_DIR/$COUNT - $NAME"
	cp -r "$VIDEOFOLDER" "$NEWDIR"
	rm -r "$NEWDIR/offline_data"
	mkdir "$NEWDIR/offline_data"
	cp "$VIDEOFOLDER/offline_data/reference_locations.msgpack" "$NEWDIR/offline_data/reference_locations.msgpack"
	old="$(pwd)"
	cd "$PUPIL_LOCATION/pupil_src"
	
	if [ "$VERBOSE" = 0 ]
	then
		python "main.py" "player" "$NEWDIR" --plugin=$PLUGIN --save-masks=$SAVE_MASKS $CUSTOMELLIPSE --min_pupil_size=$MINIMUMDIAMETER >nul 2>nul >/dev/null &
	else
		python "main.py" "player" "$NEWDIR" --plugin=$PLUGIN --save-masks=$SAVE_MASKS $CUSTOMELLIPSE --min_pupil_size=$MINIMUMDIAMETER &
	fi
	
	PROCESS=$!
	sleep 14
	while [ ! -f "$NEWDIR/offline_data/offline_pupil.meta" ]
	do
		sleep 0.2
	done
	# kill -9 $PROCESS
	# sleep 5
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
		run_pupil "${ADDR[0]}" "${ADDR[1]}" "${ADDR[2]}" "$count" "${ADDR[3]}" "${ADDR[4]}" "${ADDR[5]}" "${ADDR[6]}"
		IFS="$oIFS"
		echo "$count / $total"
		[[ "$line" == *","* ]] && ((++count))
	done
} < $IN_FILE; \

IFS="," read -ra ADDR <<< "$line"
run_pupil "${ADDR[0]}" "${ADDR[1]}" "${ADDR[2]}" "$count" "${ADDR[3]}" "${ADDR[4]}" "${ADDR[5]}" "${ADDR[6]}"
echo "$count / $total"
echo "done"
sleep infinity