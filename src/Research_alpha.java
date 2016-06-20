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
public class Research_alpha extends Research{
    private double nabla;
    private double interval;
    
    private NeuralNetwork network;

    public Research_alpha(String folder, int L, double nabla, int n, double interval) throws IOException {
        this.nabla = nabla;
        this.interval = interval;
        
        getSets(new File(folder),n);
        
        network = this.network(L, interval);
    }
    
    public void research(double [] alphas){
        for(double alpha : alphas){
            learn(alpha);
        }
    }
    
    private void learn(double alpha){
        int m = 10;
        double [] ts = new double[m];
        double [] es = new double[m];
        
        for(int i=0; i<m; i++){
            ts[i] = network.learn(trainingSet, nabla, alpha);
            es[i] = error(network, validationSet);
            network.resetWeights(interval);
        }
        
        System.out.println(alpha + " " + super.avg(ts) + " " + super.avg(es));
    }
}
