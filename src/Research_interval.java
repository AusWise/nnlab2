
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
public class Research_interval extends Research{
    private double nabla;
    private double alpha;
    private int n;
    
    private NeuralNetwork network;

    public Research_interval(String folder, int L, double nabla, double alpha, int n) throws IOException {
        this.nabla = nabla;
        this.alpha = alpha;
        this.n = n;
        
        network = super.network(L, nabla);
        super.getSets(new File(folder), n);
    }
    
    public void research(double [] intervals){
        for(double interval : intervals){
            learn(interval);
            
        }
    }
    
    private void learn(double interval){
        int m = 10;
        double [] ts = new double[m];
        double [] es = new double[m];
        
        for(int i=0; i<m; i++){
            ts[i] = network.learn(trainingSet, nabla, alpha);
            es[i] = error(network, validationSet);
            network.resetWeights(nabla);
        }
        
        System.out.println(interval + " " + super.avg(ts) + " " + super.avg(es));
    }
    
    
}
