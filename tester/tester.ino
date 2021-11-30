#define INPUT_PIN A0
// #define READ_VCC

int result = 0;

void setup() {
    Serial.begin(115200);
}

void loop() {
    #ifndef READ_VCC
    int prompt = Serial.read();
    if (prompt != -1) {
        result = analogRead(INPUT_PIN);
        Serial.println(result);
    } // else Serial.println(prompt);
    #else
    result = read_vcc();
    Serial.println(result);
    #endif
    // Note: actual VCC reading with a voltmeter: 4.613 V
    // Read VCC = 4663.08251953125±9.619952364524071, based on 2048 measurements
    // Read VCC = 4671.031073446327±6.897034938153386, based on 1770 measurements.
}

// Source: https://code.google.com/archive/p/tinkerit/wikis/SecretVoltmeter.wiki
long read_vcc() {
  long result;
  // Read 1.1V reference against AVcc
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  delay(2); // Wait for Vref to settle
  ADCSRA |= _BV(ADSC); // Convert
  while (bit_is_set(ADCSRA,ADSC));
  result = ADCL;
  result |= ADCH<<8;
  result = 1126400L / result; // Back-calculate AVcc in mV
  return result;
}
