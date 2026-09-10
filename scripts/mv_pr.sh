#!/bin/sh
if [ -z "$1" ]; then
    echo Specify PR number
    exit 1
fi
pr_no=$1
if [ -z "$2" ]; then
    body_file=pr_comment.md
else
    body_file=$2
fi
repo=scikit-image/scikit-image
base=pre-megamove-integration
gh pr edit $pr_no --repo $repo --base $base
gh pr comment $pr_no --repo $repo --body-file $body_file
gh pr edit $pr_no --add-label "needs-review"
