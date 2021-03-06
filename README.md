# My personal website

The website was built with Jekyll and the *Minimal Mistakes* theme (by [Michael Rose](https://mademistakes.com)) with a few tweaks stolen from [Grant McDermott](http://grantmcdermott.github.io) and [Paweł Budzianowski](http://budzianowski.github.io) and a few other tweaks of my own.


Add a submodule and a symlink pointing to the `Articles` repository 
([here](https://stackoverflow.com/a/27770463) and [here](https://stackoverflow.com/a/18712756))
```shell script
git submodule add -b master https://github.com/tomekkobrak/Articles
git mv Articles _posts
# and update:
git submodule update --rebase --remote
# and commit changes
git add _posts
```
