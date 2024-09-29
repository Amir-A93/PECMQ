#!/bin/zsh  


sshpass -p "Ar19930607Ar" scp -r -oProxyCommand="sshpass -p Ar1993.0607Ar ssh -W %h:%p laapour@logti-kirchhoff.ens.ad.etsmtl.ca" pecmq_extended aapour@aapour-01:~
