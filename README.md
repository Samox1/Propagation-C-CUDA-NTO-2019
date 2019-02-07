# Nowoczesne Technologie Obliczeniowe - 2019
## Projekt CUDA - propagacja światła za przeźroczem - model 3D

Projekt opierający się o komunikację MPI i bibliotekę CUDA, by wykorzystać pełny potencjał obliczeniowy klastra DWARF wydziału Fizyki PW. <br />

Program w skrócie wykonuje się w następujących krokach:
1.  Stworzenie dwuwymiarowej tablicy o wymiarach zadanych przez użytkownika (tablica ta jest przezroczem, przez które będzie propagować się światło).
2.  Wpisanie tablicy utworzonej w punkcie 1 do dwa razy większej tablicy (mniejsza tablica umieszczona w środku większej).
3.  Nowo utworzona tablica (od teraz nazywana tablicą wejściową) kopiowana jest na kartę graficzną (GPU).
4.  Na GPU liczona jest szybka transformata Fouriera (FFT) tablicy wejściowej.
5.  Wynik wysyłany jest do procesów MPI, który następnie kopiowany jest na GPU.
6.  Każdy proces oblicza dwuwymiarową tablicę odpowiedzi impulsowej (h(z) - zależna od odległości propagacji z) i wysyła ją na GPU.
7.  GPU obliczają FFT odpowiedzi impulsowej, a następnie mnożą transformatę tablicy wejściowej z transformatą odpowiedzi impulsowej.
8.  Wynik mnożenia zostaje poddany odwrotnej transformacie Fouriera. Po wykonaniu odwrotnej transformaty, jej wynik kopiowany jest z GPU na hosta.
9.  Otrzymana tablica danych zostaje poddana operacji ROLL. 
10. Ostateczny wynik zostaje zapisany do pliku.

* [x] Wybór wydajnego algorytmu

Jako metodę symulacji propagacji światła wykorzystana została metoda splotowa. Bardzo dobrą literaturą ukazującą algorytm, jak i zalety i wady tej metody jest artykuł profesora Macieja Sypka [[1]](https://gitlab.com/SimonPW/nto-2019/blob/master/B_01_199504_OptComm.PDF). Ze względu na czasochłonne obliczenia związane ze splotem dwóch funkcji, w tym przypadku tablicy wejściowej (u<sub>1</sub>) oraz odpowiedzi impulsowej (*h*) - PSF (Point Spread Function), wykorzystano własności transformacji Fouriera. Szybszą i wydajniejszą metodą będzie zrobienie *FFT{u<sub>1</sub>}* oraz *FFT{h}* i wymnożenie obu tablic ze sobą, *U<sub>2</sub>* = *FFT{u<sub>1</sub>}* x *FFT{h}*. Wynikiem odwrotnej transormaty Fouriera  tablicy *U<sub>2</sub>* , będzie tablica zespolonych wartości, *u<sub>2</sub>* , które zawierają informację o amplitudzie i fazie w danym miejscu w przestrzeni. <br />


* [x] Analiza i optymalizacja algorytmu

Algorytm wykonania propagacji wzdłuż osi *z* wydaje się dosyć trywialny. Można pokusić się o analityczne rozwiązanie tranformacji Fouriera funkcji PSF, by pominąć wykonanie jednej tranformacji Fouriera. Nie rzutowałoby to na jakość obliczeń a przyśpieszenie byłoby zauważalne. 
Jedną z wad tego algorytmu są krawędzie tablic - wprowadzające zniekształcenia i wysokoczęstotliwościowy szum. Aby pominąć szkodliwe efekty *FFT* na krawędziach proponowana jest modyfikacja algorytmu w postaci powiększenia tablic obliczeniowych do wymiarów *2Nx2N*. Tablica wejściowa, *NxN* , w tym przypadku byłaby przepisywana na środek większej tablicy. Tablicą wyjściową była by środkowa część *NxN*.

* [x] Przykładowy kod w Pythonie

Wykonano przykładowy kod w Pythonie wykorzystujący biblioteki numpy, scipy, PIL i pathlib. Używając tablicy 1024x1024 piksele sprawdzono amplitudę i fazę po propagacji na odległość 1 metra. Wynik wydaje się być prawidłowy porównując do obrazów z programu LightSword stworzonego specjalnie do takich obliczeń.
Optymalizacji może jeszcze ulec funkcja PSF by uzyskać od razu *FFT{h}*. Działający algorytm jest podstawą do wyboru kierunku obliczeń przyszłego programu.

* [x] Przegląd Literatury

Propagacja światła koherentnego przez dowolne przeźrocze - dyfrakcja - jest ciągle ciekawym tematem dla naukowców - pomaga zrozumieć naturę światła i jest szeroko wykorzystywana podczas symulacji układów optycznych. 
Istnieje kilka metod symulacji propagacji światła m.in. liczenie całki tablicy wejściowej oraz metoda splotowa.

## Wybór rodzaju obliczeń: 
* [ ] Obliczenia na tablicach o dużych wartościach *N* (duża precyzja lub duża powierzchnia przeźrocza)
* [ ] Obliczenia wzdłuż osi *Z* (duża odległość lub duża dokładność)
____________________________________________________________________________________________________________

* [ ] Implementacja algorytmu w C++

* [ ] Optymalizacja kodu pod kątem CUDA

* [ ] Wykorzystanie MPI i CUDA

* [ ] Docelowa architektura wykorzystująca protokół MPI i wiele GPU na jednym węźle


