$ver=0.0.4
mv build/exe.win-amd64-3.3 build/evilometer_exe-$ver
tar -C build -cv evilometer_exe-$ver -f build/evilometer_exe-$ver.tar
lzip -k build/evilometer_exe-$ver.tar
