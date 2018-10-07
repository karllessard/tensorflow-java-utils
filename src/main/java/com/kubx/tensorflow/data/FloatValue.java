package com.kubx.tensorflow.data;

import java.nio.FloatBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class FloatValue extends Value<FloatValue, FloatBuffer> {
 
  public static FloatValue of(Operand<Float> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static FloatValue of(Output<Float> output) {
    return of((Tensor<Float>)output.tensor());
  }

  public static FloatValue of(Tensor<Float> t) {
    return of(t.buffer().asFloatBuffer(), t.shape());
  }

  public static FloatValue of(FloatBuffer buffer, long[] shape) {
    return new FloatValue(buffer, 0, toIndices(shape));
  }
  
  public float scalar() {
    if (indices.size() > 0) {
      throw new IllegalArgumentException("Cannot convert value of " + indices.size() + " dimensions to scalar");
    }
    return buffer.get(0);
  }
  
  @Override
  protected FloatValue newValue(FloatBuffer buffer, int position, List<Index> indices) {
    return new FloatValue(buffer, position, indices);
  }
  
  @Override
  protected FloatBuffer slice() {
    return buffer.slice();
  }
  
  private FloatValue(FloatBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
