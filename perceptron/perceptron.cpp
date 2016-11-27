#include "perceptron.h"
#include <set>
#include <ctime>
Perceptron::Perceptron(arma::mat x, arma::vec y)
{
	w = arma::randu<arma::vec>(x.n_cols);
	Y=y;
	X=x;
	alpha = 0.01;
	iteration = 10000;
	sample();
	std::cout<<"Initial weights: \n";
	w.print();
	std::cout << std::endl;
}
void Perceptron::sample()
{
	
	srand(time(NULL));
	int cnt =  X.n_rows * 0.3; //30 % of the data will be used as test cases;
    int k = 0;
    std::set<int> s;

    while(k < cnt)
    {
    	int r = rand()/double(RAND_MAX) * X.n_rows;
    	if (!s.count(r))
    	{
    		s.insert(r);
    		testingX.insert_rows(0,X.row(r));
    		testingY.insert_rows(0,Y.row(r));
    		k++;
    	}
    }
    for(int i = 0; i < (int)X.n_rows; i++)
    {
    	if (!s.count(i)){
    		trainingX.insert_rows(0, X.row(i));
    		trainingY.insert_rows(0, Y.row(i));
    	}

   	}

    
}
void Perceptron::setAlpha(int a){	alpha = a;}
void Perceptron::setIteration(int a) {iteration = a;}

bool Perceptron::train()
{
  /* we are going to train on 70% of the dataset and */
  /* we are only going to run the training for user specified num of iterations, if we still cannot find a optimal weight, we will
  tell user that the dataset is not linearly seperatable.*/
    int num_cases = trainingY.n_rows; 
	
	/**** training state ****/
	for(int k = 0; k < iteration; k++)
	for(int i = 0; i < num_cases; i++)
	{  
		arma::mat t = trainingX.row(i).t();//an entry of data from x1 ... xn
		arma::mat k = w%t;
		double res= arma::accu(k); // dot product between w and t;
		double decision;
		if (res < 0) 
			decision = -1;
		else
			decision = 1;

		if (decision == trainingY(i) ) {
		  continue; // correctly predict
		}
		/*perceptron learning using the conecpet that if the decision is not matching with the actual, we update it with 
          wj =  wj + (learninng rate) * (training feature j) * ((target output) - (hypothesis output)) 
		 */
		for(int j = 0; j < (int)trainingX.n_cols; j++)
		{
			w(j) = w(j) + alpha * t(j) * (trainingY(i)-res); 
		}
	
	}

	/**** testing stage -- verify if it's linerly sepearable ****/
	int flag = 1;
	for(int i = 0; i < num_cases; i++)
	{  
		arma::mat t = trainingX.row(i).t();//an entry of data from x1 ... xn
		arma::mat k = w%t;
		double res= arma::accu(k); // dot product between w and t;
		double decision;
		if (res < 0) 
			decision = -1;
		else
			decision = 1;

		if (decision == trainingY(i) ) continue; // correctly predict
		flag = 0;
		break;
	}
	std::cout << "Final weights: \n";
	w.print();

	if (flag)
	return true;
		
	return false;
  
}

double Perceptron::test()
{
	/**** testing stage -- verify if our weights work on the testing dataset****/
	int num_cases = testingY.n_cols;
	int cnt = 0;
	double percent = 1.0/num_cases;
	for(int i = 0; i < num_cases; i++)
	{  
		arma::mat t = testingX.row(i).t();//an entry of data from x1 ... xn
		arma::mat k = w%t;
		double res= arma::accu(k); // dot product between w and t;
		double decision;
		if (res < 0) 
			decision = -1;
		else
			decision = 1;

		if (decision == testingY(i) ) {
		 cnt ++;
		}
	
	}

	return cnt *100.0* percent;
}