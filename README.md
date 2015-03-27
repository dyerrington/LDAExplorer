# LDAExplorer

This is a utility class I wrote for my Rapstats project.  One day I may build a nice UI around it so one could easily load in new corpus and browse the data.  I found that I couldn't really get a good sense of what LDA was doing with a large set of data, without being able to quickly navigate all the documents related to a topic for QA.

In any case, the easiest way to test this out is to put your documents into a subdirectory within "data", then run:

```
python lib/LDAExplorer.py
```

The default number of topics is set to 3.  Alpha is set to 'auto'.  Update the class attributes in the head to improve results.
