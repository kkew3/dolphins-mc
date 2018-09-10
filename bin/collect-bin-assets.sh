#!/bin/bash

#
# Collect (large) binary assets into tgz archives and upload to Google
# drive. Where to upload is passed in by environment variable
# GDCP_PARENT_ID. If it's not set, do not upload immediately.
#
# Author: Kaiwen Wu
#

CURDIR="`pwd`"

project_home=$(realpath "$(dirname $BASH_SOURCE)/..")
cd "$project_home"

# Where to store the created tarball locally
local_asset_dir=.assets
to_upload_tag=$local_asset_dir/to_upload.txt
mkdir -p $local_asset_dir

# Why choose these files?
# - *.pth: the state dictionary of models
# - *.tar: tarballs and state dictionary of models
# - *.tar.gz, *.zip: possibly large archive files
# - *.p, *.pkl, *.pickle: possibly large pickle files
#
# Why to exclude the three directories?
# - ./pytorch.rt: a local virtualenv directory on Kaiwen's computer
#   containing modules `pytorch` and `torchvision`
# - ./.data: a local repository of pytorch dataset on Kaiwen's computer
# - ./.assets: the assets directory itself
#
# Assume there are not white spaces in the extracted filenames
bin_assets_list=$(find . -type f \( -name '*.pth' -or \
	                            -name '*.tar' -or \
	                            -name '*.tar.gz' -or \
	                            -name '*.tgz' -or \
	                            -name '*.zip' -or \
	                            -name '*.p' -or \
	                            -name '*.pkl' -or \
	                            -name '*.hkl' -or \
	                            -name '*.h5' -or \
	                            -name '*.pickle' -or \
	                            -name '*.hickle' -or \
	                            -name '*.npz' -or \
				    -name '*.npy' -or \
	                            -name '*.dump' -or \
	                            -name '*.result' -or \
	                            -name '*.avi' -or \
	                            -name '*.mp4' -or \
	                            -name '*.mp3' -or \
	                            -name '*.png' -or \
	                            -name '*.jpg' -or \
	                            -name '*.jpeg' \)\
	| grep -v '^\./pytorch\.rt'\
	| grep -v '^\./\.data'\
	| grep -v '^\./\.assets')

# get the latest asset modification datetime (Unix ticks);
# and terminate if there's no asset
if [ "$(echo $bin_assets_list | wc -w)" -eq 0 ]; then
	exit 0
fi
this_script_mod=$(stat -c'%Y' $BASH_SOURCE)
bin_assets_mod=$(echo "$bin_assets_list" | tr ' ' '\n' | sed '/^$/d'\
	| xargs stat -c'%Y' | sed '$a0' | sort -nr | head -1)
if [ "$this_script_mod" -gt "$bin_assets_mod" ]; then
	bin_assets_mod=$this_script_mod
fi

# get the latest tarball modification datetime (Unix ticks);
# if there's no tarball, `tarball_mod` is 0
tarballs=$(find $local_asset_dir -type f -name '*.tar.gz')
if ls $local_asset_dir/*.tar.gz > /dev/null 2>&1; then
	tarball_mod=$(echo "$tarballs" | tr ' ' '\n' | sed '/^$/d'\
		| xargs stat -c'%Y' | sed '$a0' | sort -nr | head -1)
else
	tarball_mod=0
fi

if [ "$bin_assets_mod" -gt "$tarball_mod" ]; then
	# time is not in format "%H:%M:%S" because tar will recognize file
	# with comma in its name as a remote file, causing problem at both
	# creation and extraction
	tar_filename="$local_asset_dir/$(date +'%Y-%m-%d_%H-%M-%S').tar.gz"
	rm -f $local_asset_dir/*.tar.gz
	echo $bin_assets_list | tr ' ' '\n' | tar czf $tar_filename -T -
	echo $tar_filename > $to_upload_tag
fi

# Proceed to upload if there's something to upload
if ! ls $to_upload_tag > /dev/null 2>&1; then
	exit 0
fi
# Proceed to upload if GDCP_PARENT_ID had been set
if [ "$GDCP_PARENT_ID" = "" ]; then
	exit 0
fi
# Proceed to upload if `gdcp` has been installed
if ! which gdcp > /dev/null; then
	echo Attempting to upload but no gdcp found\; aborted
	exit 1
fi
# Upload
gdcp upload -p $GDCP_PARENT_ID `cat $to_upload_tag`
if [ "$?" = 0 ]; then
	rm $to_upload_tag
else
	exit 1
fi

cd "$CURDIR"
