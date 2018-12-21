OBJS = cpgraph.svg cpgraph.png

.PHONY: all clean

all: $(OBJS)

cpgraph.svg: cpgraph.dot
	dot -Tsvg $< -o $@
	test $(shell uname) = Darwin && open $@

cpgraph.png: cpgraph.dot
	dot -Tpng $< -o $@

clean:
	rm -f $(OBJS)
