# Zeclinics app

First of all we have to clone the repository to our machine with git clone:
```
git clone https://github.com/hugojm/zeclinics_app.git
```

or click on code and download zip.

## Linux

Once we have download the code we have to enter to the directory and install the dependences

```
cd zeclinics_app
pip3 install --upgrade --upgrade-strategy eager --user .
cd ..
rm -rf zeclinics_app
```
Now, we have the library installed, so every time we want to deploy the app we have to run in the terminal the following command:

```
zeclinicsapp
open 127.0.0.1:5000 in any browser
```

## Windows

Once we have download the code we have to enter to the directory and install the dependences

```
cd zeclinics_app
pip3 install --upgrade --upgrade-strategy eager --user .
cd ..
rmdir /s zeclinics_app
```

Check the directory where the package has been installed and run the file:

```
zeclinics.exe
```
