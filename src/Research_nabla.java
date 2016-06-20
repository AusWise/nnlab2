import java.io.File;
import java.io.IOException;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author auswise
 */
public class Research_nabla extends Research{
    private double alpha;
    private double interval;
    private NeuralNetwork network;

    public Research_nabla(String folder, int L, double alpha, int n, double interval) throws IOException {
        this.alpha = alpha;
        this.interval = interval;
        getSets(new File(folder),n);
        network = this.network(L, interval);
    }
    
    public void research(double [] nablas){
        for(double nabla : nablas){
            learn(nabla);
        }
    }
    
    private void learn(double nabla){
        int m = 10;
        double [] ts = new double[m];
        double [] es = new double[m];
        
        for(int i=0; i<m; i++){
            ts[i] = network.learn(trainingSet, nabla, alpha);
            es[i] = error(network, validationSet);
            network.resetWeights(interval);
        }
        
        System.out.println(nabla + " " + super.avg(ts) + " " + super.avg(es));
    }
    
    
}
