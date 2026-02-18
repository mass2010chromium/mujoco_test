mkdir -p ~/.vim/syntax
cp pddl.vim ~/.vim/syntax/
echo "au BufRead,BufNewFile *.pddl    set filetype=pddl" >> ~/.vimrc
