from corga.domain.weather import WeatherReport, WeatherProvider
from corga.domain.timeutil import Interval


class Weather:
    """The forecast, from the configured provider: live or the fixture."""

    def __init__(self, provider: WeatherProvider):
        self.provider = provider
        self.source = provider.source
        self.label = provider.label

    def report(self, span: Interval) -> WeatherReport:
        """Fetch the forecast hours over `span` as a report.

        Args:
            span: The hours to cover.

        Returns:
            The report: the source, the range, the hours and their rain.

        Raises:
            WeatherUnavailable: The provider fails or has no hours.
        """
        hours = self.provider.hourly(span.start, span.end)

        return WeatherReport(source=self.source, label=self.label, start=span.start, end=span.end, hours=hours)
