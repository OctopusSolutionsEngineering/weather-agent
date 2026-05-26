"""Azure authentication verification at startup."""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ServiceRequestError,
)
from azure.keyvault.secrets import SecretClient
from azure.appconfiguration import AzureAppConfigurationClient

logger = logging.getLogger(__name__)


# Azure Resource Manager scope — works for any Azure resource for a basic token test
ARM_SCOPE = "https://management.azure.com/.default"
APPCONFIG_SCOPE = "https://azconfig.io/.default"
KEYVAULT_SCOPE = "https://vault.azure.net/.default"


@dataclass
class AuthCheckResult:
    """Result of an authentication check."""
    name: str
    success: bool
    duration_ms: int
    detail: str = ""
    error: Optional[str] = None


@dataclass
class AuthReport:
    """Full report of all auth checks."""
    overall_success: bool
    checks: list[AuthCheckResult] = field(default_factory=list)
    identity_info: dict = field(default_factory=dict)
    
    @property
    def failed_checks(self) -> list[AuthCheckResult]:
        return [c for c in self.checks if not c.success]
    
    def summary(self) -> str:
        lines = [
            f"{'✅' if self.overall_success else '❌'} Azure auth: "
            f"{len([c for c in self.checks if c.success])}/{len(self.checks)} checks passed"
        ]
        if self.identity_info:
            lines.append(
                f"   Identity: client_id={self.identity_info.get('client_id', '?')} "
                f"tenant_id={self.identity_info.get('tenant_id', '?')}"
            )
        for c in self.checks:
            icon = "✅" if c.success else "❌"
            lines.append(f"   {icon} {c.name} ({c.duration_ms}ms) {c.detail}")
            if c.error:
                lines.append(f"      └─ {c.error}")
        return "\n".join(lines)


class AzureAuthVerifier:
    """Verifies Azure authentication and resource access at startup."""
    
    def __init__(self, credential: Optional[DefaultAzureCredential] = None):
        self.credential = credential or DefaultAzureCredential(
            exclude_interactive_browser_credential=True,
        )
        self._identity_info: dict = {}
    
    # -------------------------------------------------------------
    # Individual checks
    # -------------------------------------------------------------
    
    def check_token_acquisition(self, scope: str = ARM_SCOPE) -> AuthCheckResult:
        """Verify we can obtain *any* token from the credential chain."""
        start = time.monotonic()
        try:
            token = self.credential.get_token(scope)
            duration = int((time.monotonic() - start) * 1000)
            
            # Decode the JWT (without verification — just to extract claims)
            self._identity_info = self._decode_token_claims(token.token)
            
            return AuthCheckResult(
                name="Token acquisition",
                success=True,
                duration_ms=duration,
                detail=f"expires in {token.expires_on - int(time.time())}s",
            )
        except ClientAuthenticationError as e:
            return AuthCheckResult(
                name="Token acquisition",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=self._format_auth_error(e),
            )
        except Exception as e:
            return AuthCheckResult(
                name="Token acquisition",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
    
    def check_key_vault(self, vault_url: str) -> AuthCheckResult:
        """Verify Key Vault is reachable and we have read access."""
        start = time.monotonic()
        try:
            client = SecretClient(vault_url=vault_url, credential=self.credential)
            # List secret properties (lighter than fetching a specific secret)
            # We use `max_page_size=1` to minimize the response
            iterator = client.list_properties_of_secrets(max_page_size=1)
            # Consume the first page to actually trigger the request
            _ = next(iterator.by_page(), None)
            
            duration = int((time.monotonic() - start) * 1000)
            return AuthCheckResult(
                name=f"Key Vault access ({vault_url})",
                success=True,
                duration_ms=duration,
                detail="list secrets OK",
            )
        except ClientAuthenticationError as e:
            return AuthCheckResult(
                name=f"Key Vault access ({vault_url})",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=self._format_auth_error(e),
            )
        except HttpResponseError as e:
            return AuthCheckResult(
                name=f"Key Vault access ({vault_url})",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"HTTP {e.status_code}: {e.reason} — check RBAC role 'Key Vault Secrets User'",
            )
        except ServiceRequestError as e:
            return AuthCheckResult(
                name=f"Key Vault access ({vault_url})",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"Network error: {e} — check vault URL / DNS / firewall",
            )
        except Exception as e:
            return AuthCheckResult(
                name=f"Key Vault access ({vault_url})",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
    
    def check_specific_secret(self, vault_url: str, secret_name: str) -> AuthCheckResult:
        """Verify we can actually read a specific required secret."""
        start = time.monotonic()
        try:
            client = SecretClient(vault_url=vault_url, credential=self.credential)
            secret = client.get_secret(secret_name)
            duration = int((time.monotonic() - start) * 1000)
            value_preview = "***" + secret.value[-4:] if secret.value else "(empty)"
            return AuthCheckResult(
                name=f"Secret '{secret_name}'",
                success=True,
                duration_ms=duration,
                detail=f"value={value_preview}",
            )
        except Exception as e:
            return AuthCheckResult(
                name=f"Secret '{secret_name}'",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
    
    def check_app_configuration(self, endpoint: str) -> AuthCheckResult:
        """Verify App Configuration is reachable and we have read access."""
        start = time.monotonic()
        try:
            client = AzureAppConfigurationClient(
                base_url=endpoint,
                credential=self.credential,
            )
            # List config settings (lightweight check)
            iterator = client.list_configuration_settings()
            _ = next(iterator.by_page(), None)
            
            duration = int((time.monotonic() - start) * 1000)
            return AuthCheckResult(
                name=f"App Configuration ({endpoint})",
                success=True,
                duration_ms=duration,
                detail="list settings OK",
            )
        except ClientAuthenticationError as e:
            return AuthCheckResult(
                name=f"App Configuration ({endpoint})",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=self._format_auth_error(e),
            )
        except HttpResponseError as e:
            hint = ""
            if e.status_code == 403:
                hint = " — check RBAC role 'App Configuration Data Reader'"
            return AuthCheckResult(
                name=f"App Configuration ({endpoint})",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"HTTP {e.status_code}: {e.reason}{hint}",
            )
        except Exception as e:
            return AuthCheckResult(
                name=f"App Configuration ({endpoint})",
                success=False,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
    
    # -------------------------------------------------------------
    # Orchestrator
    # -------------------------------------------------------------
    
    def verify(
        self,
        *,
        key_vault_url: Optional[str] = None,
        appconfig_endpoint: Optional[str] = None,
        required_secrets: Optional[list[str]] = None,
    ) -> AuthReport:
        """Run all configured checks and return a full report."""
        report = AuthReport(overall_success=True)
        
        # 1. Token acquisition (always)
        token_check = self.check_token_acquisition()
        report.checks.append(token_check)
        report.identity_info = self._identity_info
        if not token_check.success:
            # Can't continue if we can't even get a token
            report.overall_success = False
            return report
        
        # 2. App Configuration (if configured)
        if appconfig_endpoint:
            check = self.check_app_configuration(appconfig_endpoint)
            report.checks.append(check)
            if not check.success:
                report.overall_success = False
        
        # 3. Key Vault (if configured)
        if key_vault_url:
            check = self.check_key_vault(key_vault_url)
            report.checks.append(check)
            if not check.success:
                report.overall_success = False
            else:
                # Only check specific secrets if vault is reachable
                for secret_name in (required_secrets or []):
                    sc = self.check_specific_secret(key_vault_url, secret_name)
                    report.checks.append(sc)
                    if not sc.success:
                        report.overall_success = False
        
        return report
    
    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    
    @staticmethod
    def _decode_token_claims(token: str) -> dict:
        """Decode JWT claims without verification (just for diagnostics)."""
        import base64
        import json
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return {}
            # JWT base64 needs padding
            payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            return {
                "client_id": claims.get("appid") or claims.get("azp", "unknown"),
                "tenant_id": claims.get("tid", "unknown"),
                "object_id": claims.get("oid", "unknown"),
                "identity_type": claims.get("idtyp", "unknown"),
                "expires_at": claims.get("exp", 0),
            }
        except Exception:
            return {}
    
    @staticmethod
    def _format_auth_error(e: ClientAuthenticationError) -> str:
        """Make ClientAuthenticationError messages actionable."""
        msg = str(e)
        hints = []
        
        if "DefaultAzureCredential failed" in msg:
            hints.append("No credential in the chain succeeded. In K8s, check:")
            hints.append("  • Workload Identity label on pod: azure.workload.identity/use=true")
            hints.append("  • ServiceAccount annotation: azure.workload.identity/client-id")
            hints.append("  • Federated credential in Azure AD")
        if "AADSTS70021" in msg or "no matching federated identity" in msg.lower():
            hints.append("Federated credential subject doesn't match service account.")
            hints.append("Expected: system:serviceaccount:<namespace>:<sa-name>")
        if "AADSTS700016" in msg:
            hints.append("Application not found in tenant — check AZURE_CLIENT_ID.")
        
        if hints:
            return msg + "\n      " + "\n      ".join(hints)
        return msg
