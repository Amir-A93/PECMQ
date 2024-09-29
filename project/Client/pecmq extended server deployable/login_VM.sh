#!/bin/zsh  

sshpass -p "Ar19930607Ar" ssh -oProxyCommand="sshpass -p Ar1993.0607Ar ssh -R 1883:localhost:1883 -W %h:%p laapour@10.178.8.215" -R 1883:localhost:1883 aapour@aapour-01

