#!/bin/bash

## Integration tests for Web Almond against public Thingpedia
## (API, web pages)

set -e
set -x
set -o pipefail

srcdir=`dirname $0`/..
srcdir=`realpath $srcdir`

workdir=`mktemp -d almondhome-integration-XXXXXX`
workdir=`realpath $workdir`
on_error() {
    test -n "$serverpid" && kill $serverpid || true
    serverpid=
    wait

    # remove workdir after the processes have died, or they'll fail
    # to write to it
    rm -fr $workdir
}
trap on_error ERR INT TERM

oldpwd=`pwd`
cd $workdir

covidfaq run-kfserver -n covid-faq -d $srcdir/data -f faq_list_clean.csv &
serverpid=$!


# in interactive mode, sleep forever
# the developer will run the tests by hand
# and Ctrl+C
if test "$1" = "--interactive" ; then
    sleep 84600
else
    # sleep until the model is ready
    # (including downloading all the model files)
    sleep 120

    python3 $srcdir/test/integration.py
fi

kill $serverpid
serverpid=
wait

rm -rf $workdir
