import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { OtpVerification } from './otp-verification';

describe('OtpVerification', () => {
  let component: OtpVerification;
  let fixture: ComponentFixture<OtpVerification>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [OtpVerification],
      providers: [provideRouter([])]
    })
      .compileComponents();

    fixture = TestBed.createComponent(OtpVerification);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
