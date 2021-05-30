if test -f FastTree; then
    echo "FastTree binary already exists. Not doing anything."
    exit
fi
echo "Getting FastTree.c file ..."
wget http://www.microbesonline.org/fasttree/FastTree.c .
echo "Compiling FastTree ..."
# See http://www.microbesonline.org/fasttree/#Install
gcc -DNO_SSE -O3 -finline-functions -funroll-loops -Wall -o FastTree FastTree.c -lm
