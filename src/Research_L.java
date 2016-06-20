import java.io.File;
import java.io.IOException;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 * @author auswise
 */
public class Research_L extends Research{
    double nabla, alpha;
    double interval;
    
    public Research_L(String folder, double nabla, double alpha, int n, double interval) throws IOException {
        this.nabla = nabla;
        this.alpha = alpha;
        this.interval = interval;
        getSets(new File(folder),n);
    }
    
    public void research(int [] Ls){
        for(int L : Ls)
            learn(L);
    }
    
    public void learn(int L){
        int m = 10;
        double [] ts = new double[m];
        double [] es = new double[m];
        
        NeuralNetwork network  = network(L, interval);
        for(int i=0; i<m; i++){
            ts[i] = network.learn(trainingSet, nabla, alpha);
//            System.out.println(ts[i]);
            es[i] = error(network, validationSet);
            network.resetWeights(interval);
        }
        
        System.out.println(L + " " + super.avg(ts) + " " + super.avg(es));
    }
    
//    public static void main(String [] args) throws IOException{
//        int [] Ls = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
//        Research_L r = new Research_L("/home/auswise/Documents/workspace/nnlab2/train_val_set", 0.5, 0, 400, 1.0);
//        r.research(Ls);
//    }
}

