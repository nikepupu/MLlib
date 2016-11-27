#ifndef READER_H
  # define READER_H
#include <armadillo>
#include <string>
class Reader{
public:
arma::mat readCSV(std::string name);
arma::mat readTXT(std::string name);

};


#endif
