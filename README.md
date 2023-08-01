## In short

this is AI for looking throgh your CCTV video archive and deting everything you don't want to spend time on. Also it has a special mode for working on the fly, by utilizing inotify linux kernel feature for monitoring dir you define. File arrived, than processed - that's easy.

No installation, no docs - just pass --help, then read sources.

## Thoughts

It's not even the script that is ready to use, but a good starting point for you to find what will solve your problem. The problem it intended to solve was the absense of pattern recognition in the current version of [Motion](Motion-Project/motion) which is great and rock-stable CCTV system I used for more than decade and still use.

I know, [Motion-Project/motionplus](Motion-Project/motionplus) exists, but it didn't look like something ready to use, especially for [ROCKPro64](https://pine64.com/product/rockpro64-4gb-single-board-computer/), and also I don't like cpp, so maybe this script will be starting point for my own, or yours cctv system. Object detection and calculation of the related metrics is probably the most important thing in any cctv system. Pyhton isn't that fast but let's be honest - in the Motion whole its workload is almost exclusively to compress/decompress video streams, which is done by external well optimized libs. In case of this script - of course it works similar way and the libs are also lowlevel, so it isn't likely that the language choosen will be the bottleneck.

But this is a far future, right at the moment the only possible input (and output) is files, which may arrive via scp, cifs, usb hdd, magnetic tape, AX.25, whatever and trigger processing by the menas of inotify or you may start it manually. Yes, you may just use it to sort through that giant pile of videos you've been collecting for years. Very likely you need cuda-capable GPU, or may be ROCm is also works as backend - just check and give me a feedback. Also very likely you will need to fix something related to video input and output formats. I'm sorry, but I hardcoded it. Too much for the beggining. Fork to fix, send commit here or just wait and maybe someday I make up that mess. Of may be you just know how to implement better so please just say.

### Visual object metrics

But the most important thing, is that the separate path measuring algorith for for every object is working, and what is even more important - it calculates path for every obeject separately. Very much of sarcasm about everything related in comments, but in fact if performs just perfect. For the info on how to install and how to use - just see built-in help and sources of course. The only way to avoid my code is to start writing it from scratch, although if you know an alternative, let me know it too
