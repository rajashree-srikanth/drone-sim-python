
test_doc:
	cd docs; jekyll  serve --watch

clean:
	find . -name '*~' -exec rm {} \;

