#!/bin/bash

xargs -I {} md5sum {} < data.filenames.txt > data.filenames.md5
