#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <armadillo>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include "perceptron.h"
struct stat buffer;
std::vector<std::string> options;
void init()
{
  options.push_back("perceptron");
}

int main(int argc, char ** argv)
{
  init();
  std::string type, name;
  if ( argc != 3)
  {
    std::cerr << "Usage: ./main FILE_NAME OPTIONS \n";
    return 1; 
  }
  else
  {
   type = argv[2]; 
   name = argv[1];
  }
  
  if (find(options.begin(), options.end(), type) == options.end())
  {
    std::cerr << "Currently we only support the following options: \n";
    for(int i = 0; i < (int)options.size(); i++)
      std::cerr << options[i] << " ";
    std::cerr << std::endl;
    return 1;
  }
  
  /** we are assuming that the the 1 .. n-1 are features, and the last col is the output**/
  arma::mat file;
  
  if (stat(name.c_str(), &buffer) < 0)
  {
    std::cerr<< "File: "<< name << " does not exist/do not have read permission\n";
    return 1;
  }
  if (name.find(".csv") || name.find(".txt"))
    file.load(name.c_str());
  else {
    std::cerr << "file extension not supported. We currently only support txt, csv files\n";
    return 1;
  }
  if (type == "perceptron")
  {
    Perceptron p(file.cols(0, file.n_cols-2), file.col(file.n_cols-1));
    bool status = p.train();
    if (!status)
      std::cout << "The dataset is not linearly separable\n";
    else
    {
      double correct = p.test();
        std::cout << "The training weight works on  "<<correct<<" % of the testing data\n";  
    }

  }
  
  return 0;
}
