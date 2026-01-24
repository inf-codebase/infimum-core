"""
Email service for authentication-related emails.

This module provides SMTP email sending with hardcoded HTML templates
for password reset, email verification, and feedback emails.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from loguru import logger


class EmailService:
    """Service for sending authentication-related emails via SMTP.
    
    Provides methods for sending password reset, email verification,
    and feedback emails with professional HTML templates.
    
    When SMTP credentials are not configured, emails are logged instead
    of being sent (useful for development).
    
    Example:
        service = EmailService()
        await service.send_password_reset_email(
            to_email="user@example.com",
            reset_token="abc123",
            user_name="John"
        )
    """
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        frontend_url: Optional[str] = None,
        admin_email: Optional[str] = None
    ):
        """
        Initialize the email service.
        
        All parameters default to values from auto_config if not provided.
        
        Args:
            smtp_server: SMTP server host (default: smtp.gmail.com)
            smtp_port: SMTP server port (default: 587)
            smtp_username: SMTP login username
            smtp_password: SMTP login password
            from_email: Sender email address (defaults to smtp_username)
            from_name: Sender display name (default: Your App)
            frontend_url: Base URL for email links (default: http://localhost:3000)
            admin_email: Admin email for feedback (defaults to smtp_username)
        """
        self._smtp_server = smtp_server
        self._smtp_port = smtp_port
        self._smtp_username = smtp_username
        self._smtp_password = smtp_password
        self._from_email = from_email
        self._from_name = from_name
        self._frontend_url = frontend_url
        self._admin_email = admin_email
        
        logger.info(f"Email Service initialized with SMTP server: {self.smtp_server}")
    
    def _get_config(self, name: str, default: any = None) -> any:
        """Get configuration value from auto_config or return default."""
        try:
            from core.utils import auto_config
            return getattr(auto_config, name, default)
        except ImportError:
            return default
    
    @property
    def smtp_server(self) -> str:
        if self._smtp_server is not None:
            return self._smtp_server
        server = self._get_config('SMTP_SERVER', None)
        return server if server else 'smtp.gmail.com'
    
    @property
    def smtp_port(self) -> int:
        if self._smtp_port is not None:
            return self._smtp_port
        try:
            return int(self._get_config('SMTP_PORT', 587))
        except (ValueError, TypeError):
            return 587
    
    @property
    def smtp_username(self) -> str:
        if self._smtp_username is not None:
            return self._smtp_username
        return self._get_config('SMTP_USERNAME', '')
    
    @property
    def smtp_password(self) -> str:
        if self._smtp_password is not None:
            return self._smtp_password
        return self._get_config('SMTP_PASSWORD', '')
    
    @property
    def from_email(self) -> str:
        if self._from_email is not None:
            return self._from_email
        email = self._get_config('FROM_EMAIL', None)
        return email if email else self.smtp_username
    
    @property
    def from_name(self) -> str:
        if self._from_name is not None:
            return self._from_name
        name = self._get_config('FROM_NAME', None)
        return name if name else 'Your App'
    
    @property
    def frontend_url(self) -> str:
        if self._frontend_url is not None:
            return self._frontend_url
        url = self._get_config('FRONTEND_URL', None)
        return url if url else 'http://localhost:3000'
    
    @property
    def admin_email(self) -> str:
        if self._admin_email is not None:
            return self._admin_email
        email = self._get_config('ADMIN_EMAIL', None)
        return email if email else self.smtp_username
    
    async def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Send password reset email to user.
        
        Args:
            to_email: Recipient email address
            reset_token: Password reset token
            user_name: Optional user name for personalization
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            reset_link = f"{self.frontend_url}/reset-password?token={reset_token}"
            subject = "Reset Your Password"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Password Reset</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 5px; }}
                    .content {{ padding: 20px 0; }}
                    .button {{ background-color: #007bff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 20px 0; }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }}
                    .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Password Reset Request</h1>
                    </div>
                    
                    <div class="content">
                        <p>Hello{f" {user_name}" if user_name else ""},</p>
                        
                        <p>We received a request to reset your password for your account. If you made this request, please click the button below to reset your password:</p>
                        
                        <p style="text-align: center;">
                            <a href="{reset_link}" class="button">Reset Password</a>
                        </p>
                        
                        <p>Alternatively, you can copy and paste the following link into your browser:</p>
                        <p style="word-break: break-all; color: #007bff;">{reset_link}</p>
                        
                        <div class="warning">
                            <strong>Important:</strong> This link will expire in 1 hour for security reasons. If you didn't request this password reset, you can safely ignore this email.
                        </div>
                        
                        <p>If you continue to have problems, please contact our support team.</p>
                        
                        <p>Best regards,<br>The {self.from_name} Team</p>
                    </div>
                    
                    <div class="footer">
                        <p>This is an automated message. Please do not reply to this email.</p>
                        <p>If you did not request a password reset, please ignore this email or contact support if you have concerns.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            text_content = f"""
Password Reset Request

Hello{f" {user_name}" if user_name else ""},

We received a request to reset your password for your account. If you made this request, please copy and paste the following link into your browser to reset your password:

{reset_link}

Important: This link will expire in 1 hour for security reasons. If you didn't request this password reset, you can safely ignore this email.

If you continue to have problems, please contact our support team.

Best regards,
The {self.from_name} Team

---
This is an automated message. Please do not reply to this email.
If you did not request a password reset, please ignore this email or contact support if you have concerns.
            """
            
            success = await self._send_email(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
            if success:
                logger.info(f"Password reset email sent successfully to: {to_email}")
            else:
                logger.error(f"Failed to send password reset email to: {to_email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending password reset email to {to_email}: {str(e)}")
            return False
    
    async def send_verification_email(
        self,
        to_email: str,
        verification_token: str,
        user_name: Optional[str] = None
    ) -> bool:
        """
        Send email verification email to user.
        
        Args:
            to_email: Recipient email address
            verification_token: Email verification token
            user_name: Optional user name for personalization
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            verification_link = f"{self.frontend_url}/verify-email?token={verification_token}"
            subject = f"Verify your email for {self.from_name}"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Email Verification</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 5px; }}
                    .content {{ padding: 20px 0; }}
                    .button {{ background-color: #28a745; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 20px 0; }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }}
                    .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Verify Your Email</h1>
                    </div>
                    
                    <div class="content">
                        <p>Hello{f" {user_name}" if user_name else ""},</p>
                        
                        <p>Thank you for signing up! Please verify your email address by clicking the button below:</p>
                        
                        <p style="text-align: center;">
                            <a href="{verification_link}" class="button">Verify Email</a>
                        </p>
                        
                        <p>Alternatively, you can copy and paste the following link into your browser:</p>
                        <p style="word-break: break-all; color: #28a745;">{verification_link}</p>
                        
                        <div class="warning">
                            <strong>Important:</strong> This link will expire in 24 hours for security reasons.
                        </div>
                        
                        <p>If you did not create an account, you can safely ignore this email.</p>
                        
                        <p>Best regards,<br>The {self.from_name} Team</p>
                    </div>
                    
                    <div class="footer">
                        <p>This is an automated message. Please do not reply to this email.</p>
                        <p>If you did not sign up for an account, please ignore this email.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            text_content = f"""
Verify Your Email for {self.from_name}

Hello{f" {user_name}" if user_name else ""},

Thank you for signing up! Please verify your email address by copying and pasting the following link into your browser:

{verification_link}

Important: This link will expire in 24 hours for security reasons.

If you did not create an account, you can safely ignore this email.

Best regards,
The {self.from_name} Team

---
This is an automated message. Please do not reply to this email.
If you did not sign up for an account, please ignore this email.
            """
            
            success = await self._send_email(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
            if success:
                logger.info(f"Email verification sent successfully to: {to_email}")
            else:
                logger.error(f"Failed to send email verification to: {to_email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending email verification to {to_email}: {str(e)}")
            return False
    
    async def send_feedback_email(
        self,
        feedback_text: str,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None
    ) -> bool:
        """
        Send feedback email to admin.
        
        Args:
            feedback_text: The feedback content
            user_name: Optional user name
            user_email: Optional user email for follow-up
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            subject = f"New Feedback from {self.from_name}"
            
            sender_info = "Anonymous user"
            if user_name and user_email:
                sender_info = f"{user_name} ({user_email})"
            elif user_name:
                sender_info = f"{user_name} (No email provided)"
            elif user_email:
                sender_info = f"User with email: {user_email}"
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>New Feedback</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 5px; }}
                    .content {{ padding: 20px 0; }}
                    .feedback-box {{ background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 15px; margin: 20px 0; }}
                    .sender-info {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>New Feedback Received</h1>
                    </div>
                    
                    <div class="content">
                        <div class="sender-info">
                            <strong>From:</strong> {sender_info}
                        </div>
                        
                        <div class="feedback-box">
                            <h3>Feedback:</h3>
                            <p>{feedback_text.replace(chr(10), '<br>')}</p>
                        </div>
                        
                        <p>This feedback was submitted through the {self.from_name} application.</p>
                    </div>
                    
                    <div class="footer">
                        <p>This is an automated message from the {self.from_name} feedback system.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            text_content = f"""
New Feedback Received

From: {sender_info}

Feedback:
{feedback_text}

This feedback was submitted through the {self.from_name} application.

---
This is an automated message from the {self.from_name} feedback system.
            """
            
            success = await self._send_email(
                to_email=self.admin_email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )
            
            if success:
                logger.info(f"Feedback email sent successfully to admin: {self.admin_email}")
            else:
                logger.error(f"Failed to send feedback email to admin: {self.admin_email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending feedback email: {str(e)}")
            return False
    
    async def send_custom_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send a custom email with provided content.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email body
            text_content: Plain text email body (optional)
            
        Returns:
            bool: True if email sent successfully
        """
        if text_content is None:
            # Strip HTML tags for plain text version
            import re
            text_content = re.sub(r'<[^>]+>', '', html_content)
        
        return await self._send_email(
            to_email=to_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content
        )
    
    async def _send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str
    ) -> bool:
        """
        Internal method to send email via SMTP.
        
        Args:
            to_email: Recipient email
            subject: Email subject
            html_content: HTML email content
            text_content: Plain text email content
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Check if SMTP is configured
            if not self.smtp_username or not self.smtp_password:
                logger.warning("SMTP credentials not configured - email sending skipped")
                logger.info(f"Would have sent email to: {to_email}")
                logger.debug(f"Subject: {subject}")
                logger.debug(f"Email content (text): {text_content[:200]}...")
                return True  # Return True for development/testing
            
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email
            
            # Create the plain-text and HTML version of your message
            part1 = MIMEText(text_content, "plain")
            part2 = MIMEText(html_content, "html")
            
            # Add HTML/plain-text parts to MIMEMultipart message
            message.attach(part1)
            message.attach(part2)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_email, to_email, message.as_string())
            
            return True
            
        except Exception as e:
            logger.error(f"SMTP error sending email to {to_email}: {str(e)}")
            return False
    
    def test_smtp_connection(self) -> bool:
        """
        Test SMTP connection without sending an email.
        
        Returns:
            bool: True if connection successful
        """
        try:
            if not self.smtp_username or not self.smtp_password:
                logger.warning("SMTP credentials not configured")
                return False
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
            
            logger.info("SMTP connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"SMTP connection test failed: {str(e)}")
            return False


# Singleton instance management
_email_service: Optional[EmailService] = None
_service_lock = None


def _get_lock():
    """Get or create the service lock."""
    global _service_lock
    if _service_lock is None:
        import threading
        _service_lock = threading.Lock()
    return _service_lock


def get_email_service() -> EmailService:
    """Get the singleton instance of EmailService."""
    global _email_service
    
    if _email_service is None:
        lock = _get_lock()
        with lock:
            if _email_service is None:
                _email_service = EmailService()
    
    return _email_service


def reset_email_service() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _email_service
    lock = _get_lock()
    with lock:
        _email_service = None
