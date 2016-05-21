let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
inoremap <C-Space> 
imap <Nul> <C-Space>
inoremap <expr> <Up> pumvisible() ? "\" : "\<Up>"
inoremap <expr> <Down> pumvisible() ? "\" : "\<Down>"
imap <S-Tab> <Plug>SuperTabBackward
inoremap <silent> <C-Tab> =UltiSnips#ListSnippets()
map! <S-Insert> <MiddleMouse>
vnoremap  :NextArg
nnoremap  :NextArg
nnoremap  :mks!
nnoremap  :bd
snoremap <silent>  c
xnoremap <silent> 	 :call UltiSnips#SaveLastVisualSelection()
snoremap <silent> 	 :call UltiSnips#ExpandSnippetOrJump()
nnoremap <NL> :tabp
nnoremap  :tabn
snoremap  "_c
nnoremap  :wa
nnoremap  :w
nnoremap  :tab split
nnoremap  :tabclose
nnoremap  :tab split
nnoremap   zz
noremap <silent> $ g$
nnoremap ,d :YcmShowDetailedDiagnostic
nnoremap ,df :call ClangFormatFile()
nnoremap ,dd :pyf /usr/share/vim/addons/syntax/clang-format-3.6.py
nnoremap ,ct :!ctags -R .
nmap ,hp <Plug>GitGutterPreviewHunk
nmap ,hr <Plug>GitGutterRevertHunk
nmap ,hs <Plug>GitGutterStageHunk
nnoremap <silent> ,p :CtrlP
vnoremap ,cp "+p
vnoremap ,cy "+y
nnoremap ,cy "+y
nnoremap ,hl :call ToggleSearchHL()
nnoremap ,f /\<\>
nnoremap ,r :%s/\<\>/
nnoremap ,sc :call CycleSpellCheck()
nnoremap ,cp "+p
nnoremap ,cf :YcmCompleter FixIt
nnoremap ,doc :YcmCompleter GetDoc
nnoremap ,cc :YcmDiag
nnoremap ,gs :tabe %
nnoremap ,o :tabe
nnoremap ,tl :tabe
nnoremap ,ag :tabe
noremap <silent> 0 g0
nnoremap G Gzz
nnoremap <silent> H :wincmd h
nnoremap <silent> J :wincmd j
nnoremap <silent> K :wincmd k
nnoremap <silent> L :wincmd l
nnoremap N Nzz
nmap [c <Plug>GitGutterPrevHunk
nmap ]c <Plug>GitGutterNextHunk
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
noremap <silent> j gj
noremap <silent> k gk
nnoremap n nzz
nnoremap zL zMzR:call ToggleFoldComments()
nnoremap zl :call ToggleFoldComments()
noremap <F4> :FSHere
map <F7> :make -C ./build/
map <S-F7> :make clean all -C ./build/
nnoremap <C-Up> :cw
nnoremap <C-Down> :ccl
nnoremap <C-Left> :cp
nnoremap <C-Right> :cn
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(expand((exists("g:netrw_gx")? g:netrw_gx : '<cfile>')),netrw#CheckIfRemote())
snoremap <silent> <Del> c
snoremap <silent> <BS> c
snoremap <silent> <C-Tab> :call UltiSnips#ListSnippets()
nnoremap <silent> <Plug>GitGutterPreviewHunk :GitGutterPreviewHunk
nnoremap <silent> <Plug>GitGutterRevertHunk :GitGutterRevertHunk
nnoremap <silent> <Plug>GitGutterStageHunk :GitGutterStageHunk
nnoremap <silent> <expr> <Plug>GitGutterPrevHunk &diff ? '[c' : ":\execute v:count1 . 'GitGutterPrevHunk'\
nnoremap <silent> <expr> <Plug>GitGutterNextHunk &diff ? ']c' : ":\execute v:count1 . 'GitGutterNextHunk'\
nnoremap <silent> <C-PageDown> :wincmd -
nnoremap <silent> <C-PageUp> :wincmd +
vnoremap <Right> <Nop>
nnoremap <Right> <Nop>
vnoremap <Left> <Nop>
nnoremap <Left> <Nop>
vnoremap <Down> <Nop>
nnoremap <Down> <Nop>
vnoremap <Up> <Nop>
nnoremap <Up> <Nop>
map <S-Insert> <MiddleMouse>
inoremap  :NextArg
inoremap 0 :CpyParamList 0
inoremap 9 :CpyParamList 9
inoremap 8 :CpyParamList 8
inoremap 7 :CpyParamList 7
inoremap 6 :CpyParamList 6
inoremap 5 :CpyParamList 5
inoremap 4 :CpyParamList 4
inoremap 3 :CpyParamList 3
inoremap 2 :CpyParamList 2
inoremap 1 :CpyParamList 1
inoremap d :CpyParamList 1
inoremap <silent> 	 =UltiSnips#ExpandSnippetOrJump()
inoremap <expr> <NL> pumvisible() ? "\" : "\<NL>"
inoremap <expr>  pumvisible() ? "\" : "\"
inoremap <expr>  pumvisible() ? "\" : "\"
inoremap <expr>  pumvisible() ? "\" : "\"
inoremap  :wa
inoremap  :w
inoremap { {}i
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set background=dark
set backspace=indent,eol,start
set comments=sl:/*,mb:\ *,elx:\ */
set completefunc=youcompleteme#Complete
set completeopt=preview,menuone
set cpoptions=aAceFsB
set expandtab
set fileencodings=ucs-bom,utf-8,default,latin1
set guifont=Droid\ Sans\ Mono\ 10
set guioptions=agimt
set helplang=en
set laststatus=2
set mouse=a
set printoptions=paper:a4
set ruler
set runtimepath=~/.vim,~/.vim/bundle/vundle,~/.vim/bundle/FSwitch,~/.vim/bundle/paraMark,~/.vim/bundle/cyclespellcheck,~/.vim/bundle/ag.vim,~/.vim/bundle/ctrlp.vim,~/.vim/bundle/vim-fugitive,~/.vim/bundle/vim-gitgutter,~/.vim/bundle/supertab,~/.vim/bundle/ultisnips,~/.vim/bundle/vim-snippets,~/.vim/bundle/YouCompleteMe,~/.vim/bundle/DoxygenToolkit.vim,~/.vim/bundle/vim-airline,~/.vim/bundle/vim-latex,~/.vim/bundle/rust.vim,~/.vim/bundle/vim-cpp-enhanced-highlight,~/.vim/bundle/glsl.vim,/var/lib/vim/addons,/usr/share/vim/vimfiles,/usr/share/vim/vim74,/usr/share/vim/vimfiles/after,/var/lib/vim/addons/after,~/.vim/after,~/.vim/bundle/vundle/,~/.vim/bundle/vundle/after,~/.vim/bundle/FSwitch/after,~/.vim/bundle/paraMark/after,~/.vim/bundle/cyclespellcheck/after,~/.vim/bundle/ag.vim/after,~/.vim/bundle/ctrlp.vim/after,~/.vim/bundle/vim-fugitive/after,~/.vim/bundle/vim-gitgutter/after,~/.vim/bundle/supertab/after,~/.vim/bundle/ultisnips/after,~/.vim/bundle/vim-snippets/after,~/.vim/bundle/YouCompleteMe/after,~/.vim/bundle/DoxygenToolkit.vim/after,~/.vim/bundle/vim-airline/after,~/.vim/bundle/vim-latex/after,~/.vim/bundle/rust.vim/after,~/.vim/bundle/vim-cpp-enhanced-highlight/after,~/.vim/bundle/glsl.vim/after
set scrolloff=25
set shiftwidth=2
set shortmess=filnxtToOc
set showmatch
set showtabline=2
set smartindent
set splitbelow
set splitright
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc
set tabline=%!airline#extensions#tabline#get()
set tabstop=2
set termencoding=utf-8
set updatetime=2000
set window=60
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/Documents/studium/GPU\ Praktikum/gpuc-asgmnt/Assignment3
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 Assignment3/CAssignment3.cpp
badd +1 Assignment3/CHistogramTask.cpp
badd +1 Assignment3/Convolution3x3.cl
argglobal
silent! argdel *
edit Assignment3/CAssignment3.cpp
set splitbelow splitright
wincmd t
set winheight=1 winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal cindent
setlocal cinkeys=0{,0},0),:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=sO:*\ -,mO:*\ \ ,exO:*/,s1:/*,mb:*,ex:*/,://
setlocal commentstring=/*%s*/
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=youcompleteme#Complete
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
set cursorline
setlocal cursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'cpp'
setlocal filetype=cpp
endif
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
set foldmethod=syntax
setlocal foldmethod=syntax
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=croql
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal grepprg=
setlocal iminsert=2
setlocal imsearch=2
setlocal include=
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=ccomplete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal relativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=2
setlocal noshortname
setlocal smartindent
setlocal softtabstop=0
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%!airline#statusline(1)
setlocal suffixesadd=
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'cpp'
setlocal syntax=cpp
endif
setlocal tabstop=2
setlocal tags=
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
let s:l = 1 - ((0 * winheight(0) + 29) / 58)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
1
normal! 0
tabedit Assignment3/Convolution3x3.cl
set splitbelow splitright
wincmd t
set winheight=1 winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal cindent
setlocal cinkeys=0{,0},0),:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=sO:*\ -,mO:*\ \ ,exO:*/,s1:/*,mb:*,ex:*/,://
setlocal commentstring=/*%s*/
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=youcompleteme#Complete
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
set cursorline
setlocal cursorline
setlocal define=^\\s*#\\s*define
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'cpp'
setlocal filetype=cpp
endif
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
set foldmethod=syntax
setlocal foldmethod=syntax
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=croql
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal grepprg=
setlocal iminsert=2
setlocal imsearch=2
setlocal include=
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=ccomplete#Complete
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal relativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=2
setlocal noshortname
setlocal smartindent
setlocal softtabstop=0
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%!airline#statusline(1)
setlocal suffixesadd=
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'cpp'
setlocal syntax=cpp
endif
setlocal tabstop=2
setlocal tags=
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
23
normal! zo
let s:l = 23 - ((22 * winheight(0) + 29) / 58)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
23
normal! 0
tabnext 2
if exists('s:wipebuf')
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOc
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :