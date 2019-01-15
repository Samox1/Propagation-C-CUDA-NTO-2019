# Nowoczesne Technologie Obliczeniowe - 2019
## Projekt CUDA - propagacja światła za przeźroczem - model 3D

Projekt opierający się o komunikację MPI i bibliotekę CUDA, by wykorzystać pełny potencjał obliczeniowy klastra DWARF wydziału Fizyki PW. <br />
Poniżej jest propozycja listy zadań, które doprowadzą do stworzenia projektowego programu.

* [x] Przegląd Literatury

Propagacja światła koherentnego przez dowolne przeźrocze - dyfrakcja - jest ciągle ciekawym tematem dla naukowców - pomaga zrozumieć naturę światła i jest szeroko wykorzystywana podczas symulacji układów optycznych. 
Istnieje kilka metod symulacji propagacji światła m.in. liczenie całki tablicy wejściowej oraz metoda splotowa.


* [x] Wybór wydajnego algorytmu

Jako metodę symulacji propagacji światła wykorzystana zostanie metoda splotowa. Bardzo dobrą literaturą ukazującą algorytm, jak i zalety i wady tej metody jest artykuł profesora Macieja Sypka [[1]](https://gitlab.com/SimonPW/nto-2019/blob/master/B_01_199504_OptComm.PDF). Ze względu na czasochłonne obliczenia związane ze splotem dwóch funkcji, w tym przypadku tablicy wejściowej (u<sub>1</sub>) oraz odpowiedzi impulsowej (*h*) - PSF (Point Spread Function), wykorzystano własności transformacji Fouriera. Szybszą i wydajniejszą metodą będzie zrobienie *FFT{u<sub>1</sub>}* oraz *FFT{h}* i wymnożenie obu tablic ze sobą, *U<sub>2</sub>* = *FFT{u<sub>1</sub>}* x *FFT{h}*. Wynikiem odwrotnej transormaty Fouriera  tablicy *U<sub>2</sub>* , będzie tablica zespolonych wartości, *u<sub>2</sub>* , które zawierają informację o amplitudzie i fazie w danym miejscu w przestrzeni.


* [x] Analiza i optymalizacja algorytmu

Algorytm wykonania propagacji wzdłuż osi *z* wydaje się dosyć trywialny. Można pokusić się o analityczne rozwiązanie tranformacji Fouriera funkcji PSF, by pominąć wykonanie jednej tranformacji Fouriera. Nie rzutowałoby to na jakość obliczeń a przyśpieszenie byłoby zauważalne. 
Jedną z wad tego algorytmu są krawędzie tablic - wprowadzające zniekształcenia i wysokoczęstotliwościowy szum. Aby pominąć szkodliwe efekty *FFT* na krawędziach proponowana jest modyfikacja algorytmu w postaci powiększenia tablic obliczeniowych do wymiarów *2Nx2N*. Tablica wejściowa, *NxN* , w tym przypadku byłaby przepisywana na środek większej tablicy. Tablicą wyjściową była by środkowa część *NxN*.

* [x] Przykładowy kod w Pythonie

Wykonano przykładowy kod w Pythonie wykorzystujący biblioteki numpy, scipy, PIL i pathlib. Używając tablicy 1024x1024 piksele sprawdzono amplitudę i fazę po propagacji na odległość 1 metra. Wynik wydaje się być prawidłowy porównując do obrazów z programy LightSword stworzonego specjalnie do takich obliczeń.
Optymalizacji może jeszcze ulec funkcja PSF by uzyskać od razu *FFT{h}*.


* [ ] Implementacja algorytmu w C++

* [ ] Optymalizacja kodu pod kątem CUDA

* [ ] Wykorzystanie MPI i CUDA

* [ ] Docelowa architektura wykorzystująca protokół MPI i wiele GPU na jednym węźle


