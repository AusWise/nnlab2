/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author auswise
 */
public class DigitRecognizer extends NeuralNetwork {

    public DigitRecognizer(int L, double interval) {
        super(70 + 1, L, 10 , interval);
    }
}
