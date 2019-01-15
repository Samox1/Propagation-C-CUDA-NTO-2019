# Nowoczesne Technologie Obliczeniowe - 2019
## Projekt CUDA - propagacja światła za przeźroczem - model 3D

Projekt opierający się o komunikację MPI i bibliotekę CUDA, by wykorzystać pełny potencjał obliczeniowy klastra DWARF wydziału Fizyki PW. <br />
Poniżej jest propozycja listy zadań, które doprowadzą do stworzenia programu projektowego.

* [ ] Przegląd Literatury

Propagacja światła koherentnego przez dowolne przeźrocze - dyfrakcja - jest ciągle ciekawym tematem dla naukowców - pomaga zrozumieć naturę światła i jest szeroko wykorzystywana podczas symulacji układów optycznych. 
Istnieje kilka metod symulacji propagacji światła m.in. liczenie całki tablicy wejściowej oraz metoda splotowa.


* [ ] Wybór wydajnego algorytmu

Jako metodę symulacji propagacji światła wykorzystana zostanie metoda splotowa. Bardzo dobrą literaturą ukazującą algorytm, jak i zalety i wady tej metody jest artykuł profesora Macieja Sypka [[1]](https://gitlab.com/SimonPW/nto-2019/blob/master/B_01_199504_OptComm.PDF). Ze względu na czasochłonne obliczenia związane ze splotem dwóch funkcji, w tym przypadku tablicy wejściowej (u<sub>1</sub>) oraz odpowiedzi impulsowej (*h*) - PSF (Point Spread Function), wykorzystamy własności transformacji Fouriera. Szybszą i wydajniejszą metodą będzie zrobienie *FFT{u<sub>1</sub>}* oraz *FFT{h}* i wymnożenie obu tablic ze sobą, *U<sub>2</sub>* = *FFT{u<sub>1</sub>}* x *FFT{h}*. Wynikiem odwrotnej transormaty Fouriera  tablicy *U<sub>2</sub>* , będzie tablica *u<sub>2</sub>* zespolonych wartości, które zawierają informację o amplitudzie i fazie w danym miejscu w przestrzeni.


* [ ] Analiza i optymalizacja algorytmu

* [ ] Przykładowy kod w Pythonie

* [ ] Implementacja algorytmu w C++

* [ ] Optymalizacja kodu pod kątem CUDA

* [ ] Wykorzystanie MPI i CUDA

* [ ] Docelowa architektura wykorzystująca protokół MPI i wiele GPU na jednym węźle


