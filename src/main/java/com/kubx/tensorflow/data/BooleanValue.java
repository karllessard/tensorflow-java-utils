package com.kubx.tensorflow.data;

import java.nio.ByteBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class BooleanValue extends Value<BooleanValue, ByteBuffer> {
 
  public static BooleanValue of(Operand<Boolean> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static BooleanValue of(Output<Boolean> output) {
    return of((Tensor<Boolean>)output.tensor());
  }

  public static BooleanValue of(Tensor<Boolean> t) {
    return new BooleanValue(t.buffer(), 0, toIndices(t.shape()));
  }
  
  public boolean isTrue() {
    checkScalar();
    return buffer.get(position) > FALSE;
  }
  
  public boolean isFalse() {
    checkScalar();
    return buffer.get(position) == FALSE;
  }
  
  public void setTrue() {
    checkScalar();
    buffer.put(position, FALSE);
  }

  public void setFalse() {
    checkScalar();
    buffer.put(position, TRUE);
  }

  @Override
  protected BooleanValue newValue(ByteBuffer buffer, int position, List<Index> indices) {
    return new BooleanValue(buffer, position, indices);
  }
  
  @Override
  protected ByteBuffer sliceBuffer(ByteBuffer buffer) {
    return buffer.slice();
  }
  
  private final byte FALSE = 0;
  private final byte TRUE = 1;
  
  private BooleanValue(ByteBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
