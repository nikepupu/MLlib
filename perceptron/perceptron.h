#ifndef PERCEPTRON_H
#define PERCEPTRON_H

  #include <armadillo>
  class Perceptron{
  arma::vec Y;
  arma::mat X;
  arma::mat w;
  arma::mat trainingX, testingX;
  arma::vec trainingY, testingY;
  double alpha;
  int iteration;
  void sample();
  public: 
    Perceptron(arma::mat x, arma::vec y);
    void setAlpha(int a);
    void setIteration(int a);
    bool train();
    bool predict();
    double test();

  };

#endif
