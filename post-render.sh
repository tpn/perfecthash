#!/bin/sh

rsync -av --ignore-existing static/ docs
rsync -av --ignore-existing ../perfecthash-ui/docs/ docs/create
