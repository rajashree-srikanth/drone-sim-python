
test_doc:
	cd doc; jekyll  serve --watch

clean:
	find . -name '*~' -exec rm {} \;

