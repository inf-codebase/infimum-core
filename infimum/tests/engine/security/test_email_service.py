"""Unit tests for core.engine.security.email_service module."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import smtplib


class TestEmailService:
    """Test cases for EmailService class."""

    @pytest.fixture
    def email_service(self):
        """Create a fresh EmailService instance for each test."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            from_email="noreply@example.com",
            from_name="Test App",
            frontend_url="https://app.example.com",
            admin_email="admin@example.com"
        )

    @pytest.fixture
    def email_service_no_smtp(self):
        """Create EmailService without SMTP credentials."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="",  # No credentials
            smtp_password="",
            from_name="Test App",
            frontend_url="https://app.example.com"
        )

    def test_smtp_server_property(self, email_service):
        """Test smtp_server property returns configured value."""
        assert email_service.smtp_server == "smtp.test.com"

    def test_smtp_port_property(self, email_service):
        """Test smtp_port property returns configured value."""
        assert email_service.smtp_port == 587

    def test_smtp_username_property(self, email_service):
        """Test smtp_username property returns configured value."""
        assert email_service.smtp_username == "test@example.com"

    def test_from_email_property(self, email_service):
        """Test from_email property returns configured value."""
        assert email_service.from_email == "noreply@example.com"

    def test_from_name_property(self, email_service):
        """Test from_name property returns configured value."""
        assert email_service.from_name == "Test App"

    def test_frontend_url_property(self, email_service):
        """Test frontend_url property returns configured value."""
        assert email_service.frontend_url == "https://app.example.com"

    def test_admin_email_property(self, email_service):
        """Test admin_email property returns configured value."""
        assert email_service.admin_email == "admin@example.com"


class TestEmailServicePasswordReset:
    """Test cases for password reset email functionality."""

    @pytest.fixture
    def email_service(self):
        """Create EmailService with mocked SMTP."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            from_email="noreply@example.com",
            from_name="Test App",
            frontend_url="https://app.example.com"
        )

    @pytest.mark.asyncio
    async def test_send_password_reset_email_success(self, email_service):
        """Test successful password reset email sending."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_password_reset_email(
                to_email="user@example.com",
                reset_token="abc123",
                user_name="John"
            )
            
            assert result is True
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("test@example.com", "test_password")
            mock_server.sendmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_password_reset_email_contains_link(self, email_service):
        """Test that password reset email contains correct reset link."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            await email_service.send_password_reset_email(
                to_email="user@example.com",
                reset_token="abc123",
                user_name="John"
            )
            
            # Check that sendmail was called with content containing the reset link
            call_args = mock_server.sendmail.call_args
            email_content = call_args[0][2]  # Third argument is the message
            
            assert "https://app.example.com/reset-password?token=abc123" in email_content

    @pytest.mark.asyncio
    async def test_send_password_reset_email_without_user_name(self, email_service):
        """Test password reset email without user name."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_password_reset_email(
                to_email="user@example.com",
                reset_token="abc123"
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_send_password_reset_email_no_smtp_credentials(self, email_service):
        """Test password reset email when SMTP not configured."""
        from infimum.engine.security.email_service import EmailService
        
        service = EmailService(
            smtp_username="",
            smtp_password="",
            frontend_url="https://app.example.com"
        )
        
        # Should return True but not actually send
        result = await service.send_password_reset_email(
            to_email="user@example.com",
            reset_token="abc123"
        )
        
        assert result is True


class TestEmailServiceVerification:
    """Test cases for email verification functionality."""

    @pytest.fixture
    def email_service(self):
        """Create EmailService with configuration."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            from_email="noreply@example.com",
            from_name="Test App",
            frontend_url="https://app.example.com"
        )

    @pytest.mark.asyncio
    async def test_send_verification_email_success(self, email_service):
        """Test successful verification email sending."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_verification_email(
                to_email="user@example.com",
                verification_token="verify123",
                user_name="Jane"
            )
            
            assert result is True
            mock_server.sendmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_verification_email_contains_link(self, email_service):
        """Test that verification email contains correct link."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            await email_service.send_verification_email(
                to_email="user@example.com",
                verification_token="verify123"
            )
            
            call_args = mock_server.sendmail.call_args
            email_content = call_args[0][2]
            
            assert "https://app.example.com/verify-email?token=verify123" in email_content

    @pytest.mark.asyncio
    async def test_send_verification_email_without_user_name(self, email_service):
        """Test verification email without user name."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_verification_email(
                to_email="user@example.com",
                verification_token="verify123"
            )
            
            assert result is True


class TestEmailServiceFeedback:
    """Test cases for feedback email functionality."""

    @pytest.fixture
    def email_service(self):
        """Create EmailService with configuration."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            from_email="noreply@example.com",
            from_name="Test App",
            frontend_url="https://app.example.com",
            admin_email="admin@example.com"
        )

    @pytest.mark.asyncio
    async def test_send_feedback_email_success(self, email_service):
        """Test successful feedback email sending."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_feedback_email(
                feedback_text="Great app!",
                user_name="User",
                user_email="user@example.com"
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_send_feedback_email_to_admin(self, email_service):
        """Test that feedback email is sent to admin."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            await email_service.send_feedback_email(
                feedback_text="Great app!"
            )
            
            call_args = mock_server.sendmail.call_args
            to_email = call_args[0][1]  # Second argument is recipient
            
            assert to_email == "admin@example.com"

    @pytest.mark.asyncio
    async def test_send_feedback_email_anonymous(self, email_service):
        """Test feedback email from anonymous user."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_feedback_email(
                feedback_text="Anonymous feedback"
            )
            
            assert result is True
            
            call_args = mock_server.sendmail.call_args
            email_content = call_args[0][2]
            
            assert "Anonymous user" in email_content


class TestEmailServiceCustomEmail:
    """Test cases for custom email functionality."""

    @pytest.fixture
    def email_service(self):
        """Create EmailService with configuration."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password",
            from_email="noreply@example.com",
            from_name="Test App"
        )

    @pytest.mark.asyncio
    async def test_send_custom_email(self, email_service):
        """Test sending custom email."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_custom_email(
                to_email="user@example.com",
                subject="Custom Subject",
                html_content="<h1>Hello</h1>",
                text_content="Hello"
            )
            
            assert result is True

    @pytest.mark.asyncio
    async def test_send_custom_email_auto_text(self, email_service):
        """Test custom email with auto-generated text content."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_custom_email(
                to_email="user@example.com",
                subject="Custom Subject",
                html_content="<h1>Hello World</h1>"
                # No text_content - should auto-generate
            )
            
            assert result is True


class TestEmailServiceErrors:
    """Test cases for error handling."""

    @pytest.fixture
    def email_service(self):
        """Create EmailService with configuration."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password"
        )

    @pytest.mark.asyncio
    async def test_send_email_smtp_error(self, email_service):
        """Test email sending handles SMTP errors."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_smtp.return_value.__enter__.side_effect = smtplib.SMTPException("Connection failed")
            
            result = await email_service.send_password_reset_email(
                to_email="user@example.com",
                reset_token="abc123"
            )
            
            assert result is False

    @pytest.mark.asyncio
    async def test_send_email_authentication_error(self, email_service):
        """Test email sending handles authentication errors."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = await email_service.send_password_reset_email(
                to_email="user@example.com",
                reset_token="abc123"
            )
            
            assert result is False


class TestEmailServiceSMTPConnection:
    """Test cases for SMTP connection testing."""

    @pytest.fixture
    def email_service(self):
        """Create EmailService with configuration."""
        from infimum.engine.security.email_service import EmailService
        
        return EmailService(
            smtp_server="smtp.test.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="test_password"
        )

    def test_test_smtp_connection_success(self, email_service):
        """Test SMTP connection test succeeds."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            
            result = email_service.test_smtp_connection()
            
            assert result is True

    def test_test_smtp_connection_failure(self, email_service):
        """Test SMTP connection test fails gracefully."""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_smtp.return_value.__enter__.side_effect = Exception("Connection failed")
            
            result = email_service.test_smtp_connection()
            
            assert result is False

    def test_test_smtp_connection_no_credentials(self):
        """Test SMTP connection test with no credentials."""
        from infimum.engine.security.email_service import EmailService
        
        service = EmailService(
            smtp_username="",
            smtp_password=""
        )
        
        result = service.test_smtp_connection()
        
        assert result is False


class TestEmailServiceSingleton:
    """Test cases for singleton instance management."""

    def test_get_email_service_singleton(self):
        """Test that get_email_service returns singleton."""
        from infimum.engine.security.email_service import get_email_service, reset_email_service
        
        reset_email_service()
        
        service1 = get_email_service()
        service2 = get_email_service()
        
        assert service1 is service2
        
        reset_email_service()

    def test_reset_email_service(self):
        """Test that reset_email_service clears the singleton."""
        from infimum.engine.security.email_service import get_email_service, reset_email_service
        
        service1 = get_email_service()
        reset_email_service()
        service2 = get_email_service()
        
        assert service1 is not service2
        
        reset_email_service()
