#define INPUT_PIN A0
#define TRIGGER_PIN 2
#define BAUD_RATE 2000000

int result = 0;
int mode = 0;

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

void select_mode() {
    while (true) {
        if (Serial.available() > 0) {
            int data = Serial.read();

            // We will have to different modes of operation:
            // triggering by serial (software triggering),
            // hardware triggering.

            if (data == 's') {
                mode = 0;
                Serial.println("Using software triggering.");
                break;
            } else if (data == 'h') {
                mode = 1;
                Serial.println("Using hardware triggering.");
                break;
            }
        }
    }
}

void setup() {
    pinMode(INPUT_PIN, INPUT);
    pinMode(TRIGGER_PIN, INPUT_PULLUP);
    Serial.begin(BAUD_RATE);
}

void loop() {
    select_mode();

    if (mode == 0) {
        int data;
        while (true) {
            if (Serial.available() > 0) {
                data = Serial.read();
                if (data != -1) {
                    if (data == 'v') {
                        // Note: actual VCC reading with a voltmeter: 4.613 V
                        // Read VCC = 4663.08251953125±9.619952364524071, based on 2048 measurements
                        // Read VCC = 4671.031073446327±6.897034938153386, based on 1770 measurements.
                        result = read_vcc();
                        Serial.println(result);
                    } else if (data == 'e')  {
                        break;
                    } else {
                        result = analogRead(INPUT_PIN);
                        Serial.println(result);
                    }
                } // else Serial.println(data);
            }
        }
    } else {
        int data = 0;
        int current_level = 0;
        int previous_level = 0;
        while (true) {
            // This a mechanism to stop the while-loop.
            if (Serial.available() > 0) {
                data = Serial.read();
                if (data == 'e') break;
            }

            current_level = digitalRead(TRIGGER_PIN);

            if (previous_level == 1 && current_level == 0) {
                // Read ADC.
                result = analogRead(INPUT_PIN);
                Serial.println(result);
            }

            previous_level = current_level;
        }
    }
}
