#!bin/bash
cd $PWD/RMK_support/tests
rm -rf example_sources
mkdir example_sources 
touch example_sources/__init__.py
cp ../../examples/epperlein_short_test.py ./example_sources
cp ../../examples/solkit_mijin_thesis.py ./example_sources
python3 runner.py $@
cd $PWD